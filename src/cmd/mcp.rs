use flag_rs::{CommandBuilder, CompletionResult};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Serialize, Deserialize)]
pub struct McpServerConfig {
    pub name: String,
    pub command: String,
    pub args: Vec<String>,
    pub description: Option<String>,
    pub enabled: bool,
}

#[derive(Debug, Serialize, Deserialize, Default)]
pub struct McpConfig {
    pub servers: Vec<McpServerConfig>,
}

impl McpConfig {
    fn config_path() -> Result<PathBuf, Box<dyn std::error::Error>> {
        let home = home::home_dir().ok_or("Failed to get home directory")?;
        let config_dir = home.join(".config").join("gamecode");
        Ok(config_dir.join("mcp-servers.json"))
    }

    pub fn load() -> Result<Self, Box<dyn std::error::Error>> {
        let path = Self::config_path()?;
        if !path.exists() {
            return Ok(Self::default());
        }
        
        let content = fs::read_to_string(&path)?;
        let config: McpConfig = serde_json::from_str(&content)?;
        Ok(config)
    }

    pub fn save(&self) -> Result<(), Box<dyn std::error::Error>> {
        let path = Self::config_path()?;
        
        // Ensure directory exists
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        
        let content = serde_json::to_string_pretty(self)?;
        fs::write(&path, content)?;
        Ok(())
    }

    pub fn add_server(&mut self, server: McpServerConfig) -> Result<(), Box<dyn std::error::Error>> {
        // Check if server with same name already exists
        if self.servers.iter().any(|s| s.name == server.name) {
            return Err(format!("Server '{}' already exists", server.name).into());
        }
        
        self.servers.push(server);
        self.save()?;
        Ok(())
    }

    pub fn remove_server(&mut self, name: &str) -> Result<(), Box<dyn std::error::Error>> {
        let initial_len = self.servers.len();
        self.servers.retain(|s| s.name != name);
        
        if self.servers.len() == initial_len {
            return Err(format!("Server '{}' not found", name).into());
        }
        
        self.save()?;
        Ok(())
    }
}

pub fn register(parent: &mut flag_rs::Command) {
    let cmd = CommandBuilder::new("mcp")
        .short("Manage MCP (Model Context Protocol) servers")
        .build();
    
    parent.add_command(cmd);
    
    // Register subcommands
    let mcp_cmd = parent.find_subcommand_mut("mcp").unwrap();
    register_list(mcp_cmd);
    register_add(mcp_cmd);
    register_remove(mcp_cmd);
    register_test(mcp_cmd);
}

fn register_list(parent: &mut flag_rs::Command) {
    let cmd = CommandBuilder::new("list")
        .short("List configured MCP servers")
        .run(|_ctx| {
            let config = McpConfig::load()
                .map_err(|e| flag_rs::Error::Custom(e.to_string().into()))?;
            
            if config.servers.is_empty() {
                println!("No MCP servers configured.");
                println!("Use 'gamecode mcp add' to add a server.");
            } else {
                println!("Configured MCP servers:");
                for server in &config.servers {
                    let status = if server.enabled { "enabled" } else { "disabled" };
                    println!("  {} [{}]", server.name, status);
                    println!("    Command: {} {}", server.command, server.args.join(" "));
                    if let Some(desc) = &server.description {
                        println!("    Description: {}", desc);
                    }
                }
            }
            Ok(())
        })
        .build();
    
    parent.add_command(cmd);
}

fn register_add(parent: &mut flag_rs::Command) {
    let cmd = CommandBuilder::new("add")
        .short("Add a new MCP server")
        .long("Add a new MCP server configuration. Example: gamecode mcp add myserver /path/to/server --arg1 --arg2")
        .run(|ctx| {
            let args = ctx.args();
            if args.len() < 2 {
                return Err(flag_rs::Error::ArgumentParsing(
                    "Usage: gamecode mcp add <name> <command> [args...]".to_string()
                ));
            }
            
            let name = args[0].clone();
            let command = args[1].clone();
            let server_args = args[2..].to_vec();
            
            let server = McpServerConfig {
                name: name.clone(),
                command,
                args: server_args,
                description: None,
                enabled: true,
            };
            
            let mut config = McpConfig::load()
                .map_err(|e| flag_rs::Error::Custom(e.to_string().into()))?;
            
            config.add_server(server)
                .map_err(|e| flag_rs::Error::Custom(e.to_string().into()))?;
            
            println!("Added MCP server '{}'", name);
            Ok(())
        })
        .build();
    
    parent.add_command(cmd);
}

fn register_remove(parent: &mut flag_rs::Command) {
    let cmd = CommandBuilder::new("remove")
        .aliases(vec!["delete", "rm"])
        .short("Remove an MCP server")
        .arg_completion(|_ctx, prefix| {
            match McpConfig::load() {
                Ok(config) => {
                    let mut result = CompletionResult::new();
                    for server in config.servers {
                        if server.name.starts_with(prefix) {
                            result = result.add(server.name);
                        }
                    }
                    Ok(result)
                }
                Err(_) => Ok(CompletionResult::new()),
            }
        })
        .run(|ctx| {
            let name = ctx.args().first()
                .ok_or_else(|| flag_rs::Error::ArgumentParsing(
                    "Server name required".to_string()
                ))?;
            
            let mut config = McpConfig::load()
                .map_err(|e| flag_rs::Error::Custom(e.to_string().into()))?;
            
            config.remove_server(name)
                .map_err(|e| flag_rs::Error::Custom(e.to_string().into()))?;
            
            println!("Removed MCP server '{}'", name);
            Ok(())
        })
        .build();
    
    parent.add_command(cmd);
}

fn register_test(parent: &mut flag_rs::Command) {
    let cmd = CommandBuilder::new("test")
        .short("Test connection to an MCP server")
        .arg_completion(|_ctx, prefix| {
            match McpConfig::load() {
                Ok(config) => {
                    let mut result = CompletionResult::new();
                    for server in config.servers {
                        if server.name.starts_with(prefix) && server.enabled {
                            result = result.add(server.name);
                        }
                    }
                    Ok(result)
                }
                Err(_) => Ok(CompletionResult::new()),
            }
        })
        .run(|ctx| {
            let name = ctx.args().first()
                .ok_or_else(|| flag_rs::Error::ArgumentParsing(
                    "Server name required".to_string()
                ))?;
            
            // Load config and find server
            let config = McpConfig::load()
                .map_err(|e| flag_rs::Error::Custom(e.to_string().into()))?;
            
            let server = config.servers.iter()
                .find(|s| &s.name == name)
                .ok_or_else(|| flag_rs::Error::Custom(
                    format!("Server '{}' not found", name).into()
                ))?;
            
            if !server.enabled {
                return Err(flag_rs::Error::Custom(
                    format!("Server '{}' is disabled", name).into()
                ));
            }
            
            // Run async test
            tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async {
                    crate::mcp_client::McpClient::test_server(server).await
                        .map_err(|e| flag_rs::Error::Custom(e.to_string().into()))
                })
            })
        })
        .build();
    
    parent.add_command(cmd);
}