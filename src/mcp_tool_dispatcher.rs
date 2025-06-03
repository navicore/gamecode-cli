use anyhow::Result;
use crate::mcp_tool_registry::McpToolRegistry;
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::debug;

/// Dispatcher that handles tool calls by routing them to appropriate MCP servers
pub struct McpToolDispatcher {
    registry: Arc<Mutex<McpToolRegistry>>,
}

impl McpToolDispatcher {
    pub async fn new() -> Result<Self> {
        let registry = McpToolRegistry::new().await?;
        Ok(Self {
            registry: Arc::new(Mutex::new(registry)),
        })
    }
    
    /// Dispatch a tool call to the appropriate MCP server
    /// Returns the result as a JSON Value
    pub async fn call_tool(&self, tool_name: &str, params: Value) -> Result<Value> {
        debug!("Dispatching tool call: {}", tool_name);
        
        let registry = self.registry.lock().await;
        registry.call_tool(tool_name, params).await
    }
    
    /// Get the registry for tool listing
    pub async fn get_registry(&self) -> Arc<Mutex<McpToolRegistry>> {
        self.registry.clone()
    }
    
    /// Refresh tools from all MCP servers
    pub async fn refresh_tools(&self) -> Result<()> {
        let mut registry = self.registry.lock().await;
        registry.refresh_tools().await
    }
}