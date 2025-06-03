use anyhow::{Context as _, Result};
use crate::cmd::mcp::McpConfig;
use crate::mcp_client::McpClient;
use crate::mcp_protocol::ToolSchema;
use gamecode_backend::Tool as BackendTool;
use serde_json::{json, Value};
use std::collections::HashMap;
use tracing::{debug, info, warn};

pub struct McpToolRegistry {
    /// Map from tool name to (server_name, tool_schema)
    tools: HashMap<String, (String, ToolSchema)>,
    config: McpConfig,
    client: McpClient,
}

impl McpToolRegistry {
    pub async fn new() -> Result<Self> {
        let config = McpConfig::load()
            .map_err(|e| anyhow::anyhow!("Failed to load MCP server configuration: {}", e))?;
        
        let mut registry = Self {
            tools: HashMap::new(),
            config,
            client: McpClient::new(),
        };
        
        registry.refresh_tools().await?;
        Ok(registry)
    }
    
    /// Refresh the tool list from all enabled servers
    pub async fn refresh_tools(&mut self) -> Result<()> {
        info!("Refreshing MCP tool registry");
        self.tools.clear();
        
        for server in &self.config.servers {
            if !server.enabled {
                debug!("Skipping disabled server: {}", server.name);
                continue;
            }
            
            match self.client.list_tools(server).await {
                Ok(mcp_tools) => {
                    info!("MCP server '{}' exposes {} protocol tools", server.name, mcp_tools.len());
                    eprintln!("\nDEBUG: Raw tools from MCP server '{}':", server.name);
                    for tool in &mcp_tools {
                        eprintln!("  - name: {}", tool.name);
                        eprintln!("    description: {}", tool.description);
                        eprintln!("    schema: {}", serde_json::to_string_pretty(&tool.input_schema).unwrap_or_default());
                    }
                    
                    // Check if this server uses the meta-tool pattern (has a list_tools tool)
                    let has_list_tools = mcp_tools.iter().any(|t| t.name == "list_tools");
                    let has_run = mcp_tools.iter().any(|t| t.name == "run");
                    
                    if has_list_tools && has_run {
                        // This is a meta-tool pattern server like gamecode-mcp
                        // We need to call list_tools to get the actual tools
                        info!("Server '{}' uses meta-tool pattern, fetching actual tools...", server.name);
                        
                        match self.client.call_tool(server, "list_tools", json!({})).await {
                            Ok(result) => {
                                // Parse the response to get actual tools
                                if let Some(content) = result.get("content") {
                                    if let Some(content_array) = content.as_array() {
                                        for item in content_array {
                                            if let Some(text) = item.get("text").and_then(|t| t.as_str()) {
                                                if let Ok(parsed) = serde_json::from_str::<Value>(text) {
                                                    if let Some(tools_array) = parsed.get("tools") {
                                                        if let Ok(actual_tools) = serde_json::from_value::<Vec<Value>>(tools_array.clone()) {
                                                            info!("Found {} actual tools from server '{}'", actual_tools.len(), server.name);
                                                            
                                                            // For meta-tool pattern, we register a special handler
                                                            // that knows to use the "run" tool
                                                            for tool_value in actual_tools {
                                                                if let Ok(mut tool_def) = serde_json::from_value::<Value>(tool_value) {
                                                                    if let Some(name) = tool_def.get("name").and_then(|n| n.as_str()) {
                                                                        let desc = tool_def.get("description")
                                                                            .and_then(|d| d.as_str())
                                                                            .unwrap_or("")
                                                                            .to_string();
                                                                        
                                                                        // Create a tool schema that will use the run meta-tool
                                                                        let tool_schema = ToolSchema {
                                                                            name: name.to_string(),
                                                                            description: desc,
                                                                            input_schema: json!({
                                                                                "type": "object",
                                                                                "properties": {
                                                                                    "tool": {
                                                                                        "type": "string",
                                                                                        "const": name,
                                                                                        "description": "Tool name"
                                                                                    },
                                                                                    "params": {
                                                                                        "type": "object",
                                                                                        "description": "Tool parameters"
                                                                                    }
                                                                                },
                                                                                "required": ["tool", "params"]
                                                                            }),
                                                                        };
                                                                        
                                                                        self.tools.insert(
                                                                            name.to_string(),
                                                                            (server.name.clone(), tool_schema)
                                                                        );
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            Err(e) => {
                                warn!("Failed to get actual tools from meta-tool server '{}': {}", server.name, e);
                            }
                        }
                    } else {
                        // Regular MCP server - tools are directly exposed
                        info!("Loaded {} tools from server '{}'", mcp_tools.len(), server.name);
                        for tool in mcp_tools {
                            let tool_name = tool.name.clone();
                            if let Some((existing_server, _)) = self.tools.get(&tool_name) {
                                warn!(
                                    "Tool '{}' already registered by server '{}', skipping from '{}'",
                                    tool_name, existing_server, server.name
                                );
                            } else {
                                self.tools.insert(tool_name, (server.name.clone(), tool));
                            }
                        }
                    }
                }
                Err(e) => {
                    warn!("Failed to list tools from server '{}': {}", server.name, e);
                    // Check if this is a common error
                    let error_msg = e.to_string();
                    if error_msg.contains("No such file or directory") || error_msg.contains("cannot find") {
                        warn!("  Server binary not found. Is it installed?");
                    } else if error_msg.contains("EOF") {
                        warn!("  Server may need a tools.yaml file in the current directory");
                    }
                }
            }
        }
        
        info!("Total tools registered: {}", self.tools.len());
        Ok(())
    }
    
    /// Convert MCP tools to Bedrock format
    pub fn to_bedrock_tools(&self) -> Vec<BackendTool> {
        self.tools
            .values()
            .map(|(server_name, tool)| {
                debug!("Converting tool '{}' from server '{}'", tool.name, server_name);
                
                // Convert MCP schema to Bedrock format
                let mut input_schema = tool.input_schema.clone();
                
                // Ensure the schema has the required Bedrock structure
                if input_schema.is_object() {
                    let obj = input_schema.as_object_mut().unwrap();
                    
                    // Bedrock expects these fields
                    obj.entry("type").or_insert(json!("object"));
                    
                    // If there's a $schema field, remove it as Bedrock doesn't like it
                    obj.remove("$schema");
                }
                
                BackendTool {
                    name: format!("{}_{}", server_name, tool.name),
                    description: tool.description.clone(),
                    input_schema,
                }
            })
            .collect()
    }
    
    /// Call a tool on the appropriate MCP server
    pub async fn call_tool(&self, full_tool_name: &str, params: Value) -> Result<Value> {
        // Parse the tool name (format: "servername_toolname")
        let parts: Vec<&str> = full_tool_name.splitn(2, '_').collect();
        if parts.len() != 2 {
            anyhow::bail!("Invalid tool name format: {}", full_tool_name);
        }
        
        let server_name = parts[0];
        let tool_name = parts[1];
        
        // Find the server
        let server = self.config.servers.iter()
            .find(|s| s.name == server_name)
            .ok_or_else(|| anyhow::anyhow!("Server '{}' not found", server_name))?;
        
        if !server.enabled {
            anyhow::bail!("Server '{}' is disabled", server_name);
        }
        
        // Call the tool
        info!("Calling tool '{}' on server '{}'", tool_name, server_name);
        self.client.call_tool(server, tool_name, params).await
    }
    
    /// Get tool info by name
    pub fn get_tool(&self, tool_name: &str) -> Option<&(String, ToolSchema)> {
        self.tools.get(tool_name)
    }
    
    /// List all available tools
    pub fn list_tools(&self) -> Vec<String> {
        self.tools.keys().cloned().collect()
    }
}