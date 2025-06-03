use anyhow::{Context, Result};
use crate::cmd::mcp::McpServerConfig;
use crate::mcp_protocol::{McpConnection, ToolSchema};
use serde_json::{json, Value};
use tracing::{debug, info, error};

// For now, we'll use a simpler approach without storing connections
// Each operation will create a new connection
pub struct McpClient;

impl McpClient {
    pub fn new() -> Self {
        Self {}
    }
    pub async fn test_server(server: &McpServerConfig) -> Result<()> {
        println!("Testing MCP server '{}'...", server.name);
        println!("Command: {} {}", server.command, server.args.join(" "));
        
        // Start the MCP server
        let process = Self.start_mcp_server(server).await?;
        let mut connection = McpConnection::new(process)?;
        
        // Initialize the connection
        println!("\nInitializing MCP connection...");
        match connection.initialize().await {
            Ok(response) => {
                println!("✓ Successfully initialized MCP connection");
                debug!("Initialize response: {:?}", response);
                
                // Send initialized notification as per MCP spec
                if let Err(e) = connection.send_notification("notifications/initialized", json!({})).await {
                    eprintln!("DEBUG: Failed to send initialized notification: {}", e);
                }
            }
            Err(e) => {
                error!("Failed to initialize: {}", e);
                return Err(e);
            }
        }
        
        // List available tools
        println!("\nQuerying available tools...");
        match connection.list_tools().await {
            Ok(tools) => {
                if tools.is_empty() {
                    println!("No tools available from this server.");
                } else {
                    println!("\nAvailable MCP tools:");
                    for tool in &tools {
                        println!("  - {}: {}", tool.name, tool.description);
                        
                        // If this is the list_tools tool, call it to see actual tools
                        if tool.name == "list_tools" {
                            println!("\nQuerying actual tools from tools.yaml...");
                            match connection.call_tool("list_tools", json!({})).await {
                                Ok(result) => {
                                    // eprintln!("DEBUG: list_tools raw result: {:?}", result);
                                    
                                    // The result might be wrapped in a content array
                                    if let Some(content) = result.get("content") {
                                        if let Some(content_array) = content.as_array() {
                                            for item in content_array {
                                                if let Some(text) = item.get("text").and_then(|t| t.as_str()) {
                                                    // eprintln!("DEBUG: Parsing text content: {}", text);
                                                    // Parse the JSON text
                                                    if let Ok(parsed) = serde_json::from_str::<Value>(text) {
                                                        if let Some(tools_array) = parsed.get("tools") {
                                                            println!("\nTools defined in tools.yaml:");
                                                            if let Some(tools) = tools_array.as_array() {
                                                                if tools.is_empty() {
                                                                    println!("  (No tools found - check if tools.yaml exists in current directory)");
                                                                    eprintln!("DEBUG: Current directory: {:?}", std::env::current_dir());
                                                                } else {
                                                                    for tool in tools {
                                                                        if let Some(name) = tool.get("name") {
                                                                            let desc = tool.get("description")
                                                                                .and_then(|d| d.as_str())
                                                                                .unwrap_or("No description");
                                                                            println!("  - {}: {}", name, desc);
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
                                },
                                Err(e) => {
                                    eprintln!("Failed to call list_tools: {}", e);
                                }
                            }
                        }
                    }
                }
            }
            Err(e) => {
                error!("Failed to list tools: {}", e);
                println!("\n✗ Failed to list tools: {}", e);
            }
        }
        
        println!("\n✓ MCP server test completed successfully");
        Ok(())
    }
    
    async fn start_mcp_server(&self, server: &McpServerConfig) -> Result<tokio::process::Child> {
        debug!("Starting MCP server: {}", server.name);
        
        use tokio::process::Command;
        let mut cmd = Command::new(&server.command);
        for arg in &server.args {
            cmd.arg(arg);
        }
        
        // Set up stdio pipes for communication
        cmd.stdin(std::process::Stdio::piped());
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());
        
        // Set working directory to current directory so tools.yaml can be found
        let cwd = std::env::current_dir().context("Failed to get current directory")?;
        debug!("Starting MCP server in directory: {:?}", cwd);
        cmd.current_dir(&cwd);
        
        let child = cmd.spawn()
            .context("Failed to spawn MCP server process")?;
            
        Ok(child)
    }
    
    pub async fn call_tool(
        &self,
        server: &McpServerConfig,
        tool_name: &str,
        params: Value,
    ) -> Result<Value> {
        info!("Calling tool '{}' on server '{}'", tool_name, server.name);
        
        // Start the MCP server and create connection
        let process = self.start_mcp_server(server).await?;
        let mut connection = McpConnection::new(process)?;
        
        // Initialize the connection
        connection.initialize().await
            .context("Failed to initialize MCP connection")?;
        
        // Send initialized notification
        let _ = connection.send_notification("notifications/initialized", json!({})).await;
        
        // Call the tool
        match connection.call_tool(tool_name, params).await {
            Ok(result) => Ok(result),
            Err(e) => {
                error!("Failed to call tool '{}': {}", tool_name, e);
                Err(e)
            }
        }
    }
    
    pub async fn list_tools(&self, server: &McpServerConfig) -> Result<Vec<ToolSchema>> {
        debug!("Listing tools from server: {}", server.name);
        
        // Start the MCP server and create connection
        let process = self.start_mcp_server(server).await?;
        let mut connection = McpConnection::new(process)?;
        
        // Initialize the connection
        connection.initialize().await
            .context("Failed to initialize MCP connection")?;
        
        // Send initialized notification as per MCP spec
        let _ = connection.send_notification("notifications/initialized", json!({})).await;
        
        // Get the list of tools
        connection.list_tools().await
    }
}