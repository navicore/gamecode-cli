use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, ChildStderr};
use tracing::{debug, info};

#[derive(Debug, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    jsonrpc: String,
    id: Value,
    method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    params: Option<Value>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    jsonrpc: String,
    id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct JsonRpcError {
    code: i32,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<Value>,
}

pub struct McpConnection {
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    stderr: Option<ChildStderr>,
    _process: Child,
    request_id: u64,
}

impl McpConnection {
    pub fn new(mut process: Child) -> Result<Self> {
        let stdin = process.stdin.take()
            .context("Failed to get stdin from MCP process")?;
        let stdout = process.stdout.take()
            .context("Failed to get stdout from MCP process")?;
        let stderr = process.stderr.take();
        
        Ok(Self {
            stdin,
            stdout: BufReader::new(stdout),
            stderr,
            _process: process,
            request_id: 0,
        })
    }
    
    pub async fn initialize(&mut self) -> Result<Value> {
        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: json!(self.next_id()),
            method: "initialize".to_string(),
            params: Some(json!({
                "protocolVersion": "0.1.0",
                "capabilities": {
                    "tools": {}
                },
                "clientInfo": {
                    "name": "gamecode-cli",
                    "version": "0.2.0"
                }
            })),
        };
        
        self.send_request(&request).await
    }
    
    pub async fn list_tools(&mut self) -> Result<Vec<ToolSchema>> {
        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: json!(self.next_id()),
            method: "tools/list".to_string(),
            params: Some(json!({})),  // Empty params object instead of None
        };
        
        let response = self.send_request(&request).await?;
        
        // Parse the response to extract tools
        if let Some(tools) = response.get("tools") {
            let tools: Vec<ToolSchema> = serde_json::from_value(tools.clone())?;
            Ok(tools)
        } else {
            Ok(vec![])
        }
    }
    
    pub async fn call_tool(&mut self, name: &str, arguments: Value) -> Result<Value> {
        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: json!(self.next_id()),
            method: "tools/call".to_string(),
            params: Some(json!({
                "name": name,
                "arguments": arguments
            })),
        };
        
        self.send_request(&request).await
    }
    
    async fn send_request(&mut self, request: &JsonRpcRequest) -> Result<Value> {
        // Send request
        let request_str = serde_json::to_string(request)?;
        debug!("Sending MCP request: {}", request_str);
        // eprintln!("DEBUG: Sending MCP request: {}", request_str);
        
        self.stdin.write_all(request_str.as_bytes()).await?;
        self.stdin.write_all(b"\n").await?;
        self.stdin.flush().await?;
        
        // Read response
        let mut response_line = String::new();
        let bytes_read = self.stdout.read_line(&mut response_line).await?;
        
        // eprintln!("DEBUG: Read {} bytes", bytes_read);
        // eprintln!("DEBUG: Received MCP response: {:?}", response_line);
        debug!("Received MCP response: {}", response_line);
        
        if response_line.is_empty() {
            anyhow::bail!("Empty response from MCP server");
        }
        
        let response: JsonRpcResponse = serde_json::from_str(&response_line)?;
        
        if let Some(error) = response.error {
            anyhow::bail!("MCP error: {} - {}", error.code, error.message);
        }
        
        response.result.context("No result in MCP response")
    }
    
    fn next_id(&mut self) -> u64 {
        self.request_id += 1;
        self.request_id
    }
    
    pub async fn send_notification(&mut self, method: &str, params: Value) -> Result<()> {
        let notification = json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params
        });
        
        let notification_str = serde_json::to_string(&notification)?;
        // eprintln!("DEBUG: Sending notification: {}", notification_str);
        
        self.stdin.write_all(notification_str.as_bytes()).await?;
        self.stdin.write_all(b"\n").await?;
        self.stdin.flush().await?;
        
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSchema {
    pub name: String,
    pub description: String,
    #[serde(rename = "inputSchema")]
    pub input_schema: Value,
}

impl Drop for McpConnection {
    fn drop(&mut self) {
        // The Child process will be killed when dropped
        info!("Closing MCP connection");
    }
}