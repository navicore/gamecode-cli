use anyhow::{Context as AnyhowContext, Result};
use flag_rs::{Command, CommandBuilder, CompletionResult, Context, Flag, FlagType, FlagValue, Shell};
use gamecode_backend::{
    BackendStatus, ChatRequest, ContentBlock, InferenceConfig, LLMBackend,
    Message as BackendMessage, MessageRole as BackendMessageRole, RetryConfig, StatusCallback,
    Tool as BackendTool,
};
use gamecode_bedrock::BedrockBackend;
use gamecode_context::{
    session::{Message as ContextMessage, MessageRole as ContextMessageRole, MessageRole},
    SessionManager,
};
use gamecode_prompt::PromptManager;
use gamecode_tools::{create_bedrock_dispatcher_with_schemas, schema::ToolSchemaRegistry};
use serde_json::{json, Value};
use std::io::Write;
use std::sync::Arc;
use std::time::Duration;
use tracing::debug;
use uuid::Uuid;

// Backend factory function to create the appropriate backend
async fn create_backend(region: &str) -> Result<Box<dyn LLMBackend>> {
    // For now, we only support Bedrock, but this could be expanded
    // to support other backends (OpenAI, etc.) based on configuration
    let backend = BedrockBackend::new_with_region(region)
        .await
        .context("Failed to create backend")?;
    Ok(Box::new(backend))
}

// Model mapping function
fn map_model_name(model: &str) -> String {
    let mapped = match model {
        "opus-4" => "us.anthropic.claude-opus-4-20250514-v1:0",
        "sonnet-4" => "us.anthropic.claude-sonnet-4-20250514-v1:0",
        "claude-3.7-sonnet" => "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        "claude-3.5-sonnet" => "anthropic.claude-3-5-sonnet-20240620-v1:0",
        "claude-3.5-haiku" => "anthropic.claude-3-5-haiku-20241022-v1:0",
        "claude-3-sonnet" => "anthropic.claude-3-sonnet-20240229-v1:0",
        "claude-3-haiku" => "anthropic.claude-3-haiku-20240307-v1:0",
        _ => model, // Pass through unknown model names
    };
    mapped.to_string()
}

// Helper function to convert gamecode-tools schemas to backend format
fn convert_tools_to_backend(schema_registry: &ToolSchemaRegistry) -> Result<Vec<BackendTool>> {
    let mut backend_tools = Vec::new();

    for bedrock_spec in schema_registry.to_bedrock_specs() {
        let tool = BackendTool {
            name: bedrock_spec.name,
            description: bedrock_spec.description,
            input_schema: bedrock_spec.input_schema.json,
        };
        backend_tools.push(tool);
    }

    Ok(backend_tools)
}

#[tokio::main]
async fn main() -> Result<()> {
    // Fix flag-rs bug: if GAMECODE_CLI_COMPLETE is set, copy it to GAMECODE-CLI_COMPLETE
    if let Ok(value) = std::env::var("GAMECODE_CLI_COMPLETE") {
        unsafe {
            std::env::set_var("GAMECODE-CLI_COMPLETE", value);
        }
    }
    
    let app = build_cli();
    
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(e) = app.execute(args) {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
    
    Ok(())
}

fn build_cli() -> Command {
    CommandBuilder::new("gamecode-cli")
        .short("AI-powered CLI assistant")
        .long("An experimental client for AWS Bedrock Anthropic Claude models used as an agentic assistant")
        
        // Global flags
        .flag(Flag::new("system-prompt")
            .short('s')
            .usage("System prompt to use")
            .value_type(FlagType::String))
        
        .flag(Flag::new("model")
            .short('m')
            .usage("Model to use (e.g., opus-4, claude-3.7-sonnet)")
            .value_type(FlagType::String))
            
        .flag(Flag::new("region")
            .short('r')
            .usage("AWS region")
            .value_type(FlagType::String)
            .default(FlagValue::String("us-west-2".to_string())))
            
        .flag(Flag::new("session")
            .usage("Session ID to continue")
            .value_type(FlagType::String))
            
        .flag(Flag::new("new-session")
            .usage("Start a new session")
            .value_type(FlagType::Bool)
            .default(FlagValue::Bool(false)))
            
        .flag(Flag::new("verbose")
            .short('v')
            .usage("Enable verbose output")
            .value_type(FlagType::Bool)
            .default(FlagValue::Bool(false)))
            
        .flag(Flag::new("no-tools")
            .usage("Disable tools entirely")
            .value_type(FlagType::Bool)
            .default(FlagValue::Bool(false)))
            
        .flag(Flag::new("max-retries")
            .usage("Maximum number of retry attempts")
            .value_type(FlagType::Int)
            .default(FlagValue::Int(20)))
            
        .flag(Flag::new("initial-retry-delay-ms")
            .usage("Initial retry delay in milliseconds")
            .value_type(FlagType::Int)
            .default(FlagValue::Int(500)))
        
        // Dynamic completions for system-prompt
        .flag_completion("system-prompt", |_ctx, prefix| {
            match PromptManager::new() {
                Ok(manager) => match manager.list_prompts() {
                    Ok(prompts) => {
                        let mut result = CompletionResult::new();
                        for prompt_name in prompts {
                            if prompt_name.starts_with(prefix) {
                                // Try to get prompt info for description
                                if let Ok(info) = manager.get_prompt_info(&prompt_name) {
                                    let size_kb = info.size / 1024;
                                    let desc = format!("{}KB", size_kb);
                                    // Sanitize description
                                    let sanitized_desc = desc.chars()
                                        .map(|c| if c.is_alphanumeric() || c.is_whitespace() { c } else { '_' })
                                        .collect::<String>();
                                    result = result.add_with_description(
                                        prompt_name,
                                        sanitized_desc
                                    );
                                } else {
                                    result = result.add(prompt_name);
                                }
                            }
                        }
                        Ok(result)
                    }
                    Err(_) => Ok(CompletionResult::new()),
                },
                Err(_) => Ok(CompletionResult::new()),
            }
        })
        
        // Dynamic completions for session
        .flag_completion("session", |_ctx, prefix| {
            match SessionManager::new() {
                Ok(manager) => match manager.list_sessions() {
                    Ok(sessions) => {
                        let mut result = CompletionResult::new();
                        for session_info in sessions {
                            let id_str = session_info.id.to_string();
                            if id_str.starts_with(prefix) {
                                // Format system time for display
                                // let created = chrono::DateTime::<chrono::Utc>::from(session_info.created_at)
                                //     .format("%Y-%m-%d %H:%M")
                                //     .to_string();
                                //let desc = format!("{} msgs created at {}", session_info.message_count, created);
                                // Sanitize description
                                // let sanitized_desc = desc.chars()
                                //     .map(|c| if c.is_alphanumeric() || c.is_whitespace() { c } else { '_' })
                                //     .collect::<String>();
                                // result = result.add_with_description(
                                //     id_str,
                                //     sanitized_desc
                                // );
                                result = result.add(
                                    //id_str,
                                    "abcdef"
                                );

                            }
                        }
                        Ok(result)
                    }
                    Err(e) => panic!("Failed to list sessions: {}", e),
                },
                Err(e) => panic!("Failed to create session manager: {}", e),
            }
        })
        
        // Dynamic completions for model
        .flag_completion("model", |_ctx, prefix| {
            // In the future, this could query the backend
            let models = vec![
                ("opus-4", "Opus 4"),
                ("claude-3.7-sonnet", "3.7 Sonnet"),
                ("claude-3.5-sonnet", "3.5 Sonnet"),
                ("claude-3.5-haiku", "3.5 Haiku"),
                ("claude-3-sonnet", "3 Sonnet"),
                ("claude-3-haiku", "3 Haiku"),
            ];
            
            let mut result = CompletionResult::new();
            for (model, desc) in models {
                if model.starts_with(prefix) {
                    // Sanitize the description - replace non-alphanumeric with underscore
                    let sanitized_desc = desc.chars()
                        .map(|c| if c.is_alphanumeric() || c.is_whitespace() { c } else { '_' })
                        .collect::<String>();
                    result = result.add_with_description(model, &sanitized_desc);
                }
            }
            Ok(result)
        })
        
        // Subcommands
        .subcommand(build_completion_command())
        .subcommand(build_models_command())
        .subcommand(build_prompts_command())
        .subcommand(build_sessions_command())
        
        // Main command handler
        .run(|ctx| {
            // Use tokio::task::block_in_place to run async code in sync context
            tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async {
                    run_main_command(ctx).await
                        .map_err(|e| flag_rs::Error::Custom(e.to_string().into()))
                })
            })
        })
        
        .build()
}

fn build_completion_command() -> Command {
    CommandBuilder::new("completion")
        .aliases(vec!["completions"])
        .short("Generate shell completion scripts")
        .long("Generate shell completion scripts for your shell")
        .arg_completion(|_ctx, prefix| {
            let shells = vec![
                ("bash", "Bourne Again Shell"),
                ("zsh", "Z Shell"),
                ("fish", "Fish Shell"),
            ];
            
            let mut result = CompletionResult::new();
            for (shell, desc) in shells {
                if shell.starts_with(prefix) {
                    // Sanitize description
                    let sanitized_desc = desc.chars()
                        .map(|c| if c.is_alphanumeric() || c.is_whitespace() { c } else { '_' })
                        .collect::<String>();
                    result = result.add_with_description(shell, &sanitized_desc);
                }
            }
            Ok(result)
        })
        .run(|ctx| {
            let shell_name = ctx.args().first()
                .ok_or_else(|| flag_rs::Error::ArgumentParsing(
                    "Shell name required (bash, zsh, or fish)".to_string()
                ))?;
            
            let shell = match shell_name.as_str() {
                "bash" => Shell::Bash,
                "zsh" => Shell::Zsh,
                "fish" => Shell::Fish,
                _ => return Err(flag_rs::Error::ArgumentParsing(
                    format!("Unsupported shell: {}", shell_name)
                )),
            };
            
            let root = build_cli();
            let completion = root.generate_completion(shell);
            // Fix flag-rs bug: replace hyphens with underscores in shell identifiers
            let fixed_completion = completion
                .replace("GAMECODE-CLI_COMPLETE", "GAMECODE_CLI_COMPLETE")
                .replace("_gamecode-cli_complete", "_gamecode_cli_complete");
            println!("{}", fixed_completion);
            
            Ok(())
        })
        .build()
}

fn build_models_command() -> Command {
    CommandBuilder::new("models")
        .short("List available models")
        .run(|_ctx| {
            println!("Available models:");
            println!("  opus-4              - Claude Opus 4 (cross-region)");
            println!("  claude-3.7-sonnet   - Claude 3.7 Sonnet (cross-region)");
            println!("  claude-3.5-sonnet   - Claude 3.5 Sonnet");
            println!("  claude-3.5-haiku    - Claude 3.5 Haiku");
            println!("  claude-3-sonnet     - Claude 3 Sonnet");
            println!("  claude-3-haiku      - Claude 3 Haiku");
            Ok(())
        })
        .build()
}

fn build_prompts_command() -> Command {
    CommandBuilder::new("prompts")
        .short("Manage prompts")
        .subcommand(
            CommandBuilder::new("list")
                .short("List available prompts")
                .run(|_ctx| {
                    let prompt_manager = PromptManager::new()
                        .map_err(|e| flag_rs::Error::Custom(Box::new(e)))?;
                    let prompts = prompt_manager.list_prompts()
                        .map_err(|e| flag_rs::Error::Custom(Box::new(e)))?;
                    
                    println!("Available prompts:");
                    for prompt_name in prompts {
                        if let Ok(info) = prompt_manager.get_prompt_info(&prompt_name) {
                            println!("  {} ({} bytes)", prompt_name, info.size);
                        } else {
                            println!("  {}", prompt_name);
                        }
                    }
                    Ok(())
                })
                .build()
        )
        .subcommand(
            CommandBuilder::new("show")
                .short("Show a specific prompt")
                .arg_completion(|_ctx, prefix| {
                    match PromptManager::new() {
                        Ok(manager) => match manager.list_prompts() {
                            Ok(prompts) => {
                                let mut result = CompletionResult::new();
                                for prompt_name in prompts {
                                    if prompt_name.starts_with(prefix) {
                                        // Try extreme sanitization on the value too
                                        let sanitized_name = prompt_name.chars()
                                            .map(|c| if c.is_alphanumeric() { c } else { '_' })
                                            .collect::<String>();
                                        eprintln!("DEBUG: Original: '{}', Sanitized: '{}'", prompt_name, sanitized_name);
                                        result = result.add(sanitized_name);
                                    }
                                }
                                Ok(result)
                            }
                            Err(_) => Ok(CompletionResult::new()),
                        },
                        Err(_) => Ok(CompletionResult::new()),
                    }
                })
                .run(|ctx| {
                    let name = ctx.args().first()
                        .ok_or_else(|| flag_rs::Error::ArgumentParsing(
                            "Prompt name required".to_string()
                        ))?;
                    
                    let prompt_manager = PromptManager::new()
                        .map_err(|e| flag_rs::Error::Custom(Box::new(e)))?;
                    let content = prompt_manager.load_prompt(name)
                        .map_err(|e| flag_rs::Error::Custom(Box::new(e)))?;
                    
                    println!("{}", content);
                    Ok(())
                })
                .build()
        )
        .build()
}

fn build_sessions_command() -> Command {
    CommandBuilder::new("sessions")
        .short("Manage sessions")
        .subcommand(
            CommandBuilder::new("list")
                .short("List available sessions")
                .run(|_ctx| {
                    let session_manager = SessionManager::new()
                        .map_err(|e| flag_rs::Error::Custom(Box::new(e)))?;
                    let sessions = session_manager.list_sessions()
                        .map_err(|e| flag_rs::Error::Custom(Box::new(e)))?;
                    
                    println!("Available sessions:");
                    for session_info in sessions {
                        let created = chrono::DateTime::<chrono::Utc>::from(session_info.created_at)
                            .format("%Y-%m-%d %H:%M:%S")
                            .to_string();
                        println!("  {} - {} ({} messages)",
                            session_info.id,
                            created,
                            session_info.message_count
                        );
                    }
                    Ok(())
                })
                .build()
        )
        .subcommand(
            CommandBuilder::new("show")
                .short("Show session details")
                .arg_completion(|_ctx, prefix| {
                    match SessionManager::new() {
                        Ok(manager) => match manager.list_sessions() {
                            Ok(sessions) => {
                                let mut result = CompletionResult::new();
                                for session in sessions {
                                    let id_str = session.id.to_string();
                                    if id_str.starts_with(prefix) {
                                        result = result.add(id_str);
                                    }
                                }
                                Ok(result)
                            }
                            Err(_) => Ok(CompletionResult::new()),
                        },
                        Err(_) => Ok(CompletionResult::new()),
                    }
                })
                .run(|ctx| {
                    let session_id_str = ctx.args().first()
                        .ok_or_else(|| flag_rs::Error::ArgumentParsing(
                            "Session ID required".to_string()
                        ))?;
                    
                    let session_id = Uuid::parse_str(session_id_str)
                        .map_err(|e| flag_rs::Error::Custom(Box::new(e)))?;
                    
                    let mut session_manager = SessionManager::new()
                        .map_err(|e| flag_rs::Error::Custom(Box::new(e)))?;
                    let session = session_manager.load_session(&session_id)
                        .map_err(|e| flag_rs::Error::Custom(Box::new(e)))?;
                    
                    println!("Session: {}", session.id);
                    println!("Created: {}", session.created_at.format("%Y-%m-%d %H:%M:%S"));
                    println!("Messages: {}", session.messages.len());
                    
                    for (i, msg) in session.messages.iter().enumerate() {
                        println!("\n[{}] {:?}:", i + 1, msg.role);
                        println!("{}", msg.content);
                    }
                    
                    Ok(())
                })
                .build()
        )
        .subcommand(
            CommandBuilder::new("delete")
                .short("Delete a session")
                .arg_completion(|_ctx, prefix| {
                    match SessionManager::new() {
                        Ok(manager) => match manager.list_sessions() {
                            Ok(sessions) => {
                                let mut result = CompletionResult::new();
                                for session in sessions {
                                    let id_str = session.id.to_string();
                                    if id_str.starts_with(prefix) {
                                        result = result.add(id_str);
                                    }
                                }
                                Ok(result)
                            }
                            Err(_) => Ok(CompletionResult::new()),
                        },
                        Err(_) => Ok(CompletionResult::new()),
                    }
                })
                .run(|ctx| {
                    let session_id_str = ctx.args().first()
                        .ok_or_else(|| flag_rs::Error::ArgumentParsing(
                            "Session ID required".to_string()
                        ))?;
                    
                    let session_id = Uuid::parse_str(session_id_str)
                        .map_err(|e| flag_rs::Error::Custom(Box::new(e)))?;
                    
                    // SessionManager doesn't expose delete_session directly
                    // For now, we'll just print a message
                    // TODO: Add delete_session to SessionManager or use storage directly
                    println!("Session deletion not yet implemented");
                    println!("Would delete session: {}", session_id);
                    Ok(())
                })
                .build()
        )
        .build()
}

async fn run_main_command(ctx: &Context) -> Result<()> {
    // Extract prompt from remaining arguments
    let prompt_parts = ctx.args();
    if prompt_parts.is_empty() {
        return Err(anyhow::anyhow!("Prompt is required when not using a subcommand"));
    }
    
    // Extract flags
    let verbose = ctx.flag("verbose")
        .and_then(|s| s.parse::<bool>().ok())
        .unwrap_or(false);
        
    let new_session = ctx.flag("new-session")
        .and_then(|s| s.parse::<bool>().ok())
        .unwrap_or(false);
        
    let no_tools = ctx.flag("no-tools")
        .and_then(|s| s.parse::<bool>().ok())
        .unwrap_or(false);
        
    let max_retries = ctx.flag("max-retries")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(20);
        
    let initial_retry_delay_ms = ctx.flag("initial-retry-delay-ms")
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(500);
    
    let region = ctx.flag("region")
        .map(|s| s.as_str())
        .unwrap_or("us-west-2");
        
    let model = ctx.flag("model").map(|s| s.as_str());
    let system_prompt_name = ctx.flag("system-prompt").map(|s| s.as_str());
    let session_id_str = ctx.flag("session").map(|s| s.as_str());
    
    // Setup logging
    let log_level = if verbose {
        tracing::Level::DEBUG
    } else {
        tracing::Level::INFO
    };
    tracing_subscriber::fmt().with_max_level(log_level).init();
    
    // Create backend with region
    debug!("Using AWS region: {}", region);
    let backend = create_backend(region).await?;
    
    // Map model name and use default if none specified
    let selected_model = model
        .map(|m| map_model_name(m))
        .unwrap_or_else(|| "us.anthropic.claude-3-7-sonnet-20250219-v1:0".to_string());
    debug!("Using model: {}", selected_model);
    
    // Detect cross-region models
    let uses_cross_region_model = selected_model.starts_with("us.");
    
    // Setup gamecode-tools dispatcher with schema generation
    let (dispatcher, schema_registry) = if no_tools {
        eprintln!("ℹ️  Running without tools (--no-tools flag)");
        let dispatcher = gamecode_tools::jsonrpc::Dispatcher::new();
        let schema_registry = gamecode_tools::schema::ToolSchemaRegistry::new();
        (dispatcher, schema_registry)
    } else {
        // For now, always use the full tool set
        // TODO: Add minimal dispatcher when available in gamecode-tools
        if uses_cross_region_model {
            eprintln!("⚠️  Using full tool set with cross-region model. Consider using --no-tools for better performance.");
        }
        create_bedrock_dispatcher_with_schemas()
    };
    let dispatcher: Arc<gamecode_tools::jsonrpc::Dispatcher> = Arc::new(dispatcher);
    
    // Setup session management
    let mut session_manager = SessionManager::new()
        .context("Failed to create session manager")?;
    
    // Load or create session based on arguments
    let mut session = if new_session {
        debug!("Creating new session");
        session_manager.new_session()?
    } else if let Some(session_id_str) = session_id_str {
        debug!("Loading session: {}", session_id_str);
        let session_id = Uuid::parse_str(session_id_str)
            .with_context(|| format!("Invalid session ID: {}", session_id_str))?;
        session_manager
            .load_session(&session_id)
            .with_context(|| format!("Failed to load session: {}", session_id))?
    } else {
        debug!("Loading latest session");
        session_manager.load_latest()?
    };
    
    debug!("Using session: {}", session.id);
    
    // Load system prompt if this is a new session (no messages yet)
    if session.messages.is_empty() {
        let prompt_manager = PromptManager::new()
            .context("Failed to create prompt manager")?;
            
        let system_prompt = if let Some(prompt_name) = system_prompt_name {
            prompt_manager
                .load_prompt(prompt_name)
                .with_context(|| format!("Failed to load prompt '{}'", prompt_name))?
        } else if uses_cross_region_model {
            eprintln!("ℹ️  Using minimal system prompt for cross-region model (33 chars instead of 475)");
            prompt_manager
                .load_prompt("minimal")
                .context("Failed to load minimal prompt")?
        } else {
            prompt_manager
                .load_default()
                .context("Failed to load default prompt")?
        };
        
        if verbose {
            if let Some(prompt_name) = system_prompt_name {
                debug!("Using named system prompt: {}", prompt_name);
            } else {
                debug!("Using default system prompt");
            }
        }
        
        // Add system prompt to session
        let system_message = ContextMessage::new(ContextMessageRole::System, system_prompt);
        session_manager.add_message(&mut session, system_message)?;
    }
    
    // Add current user prompt to session
    let user_prompt = prompt_parts.join(" ");
    let user_message = ContextMessage::new(ContextMessageRole::User, user_prompt);
    session_manager.add_message(&mut session, user_message)?;
    
    // Convert session messages to backend format
    let mut messages = Vec::new();
    for context_msg in &session.messages {
        let role = match context_msg.role {
            ContextMessageRole::System => BackendMessageRole::System,
            ContextMessageRole::User => BackendMessageRole::User,
            ContextMessageRole::Assistant => BackendMessageRole::Assistant,
            ContextMessageRole::Tool => BackendMessageRole::User, // Tool messages treated as user context
        };
        
        let message = BackendMessage::text(role, context_msg.content.clone());
        messages.push(message);
    }
    
    // Convert tools from gamecode-tools to backend format
    let backend_tools = if no_tools {
        Vec::new()
    } else {
        convert_tools_to_backend(&schema_registry)?
    };
    
    // Create retry configuration
    let retry_config = RetryConfig {
        max_retries,
        initial_delay: Duration::from_millis(initial_retry_delay_ms),
        backoff_strategy: gamecode_backend::BackoffStrategy::Exponential { multiplier: 3 },
        verbose,
    };
    
    // Create status callback for retry/backoff feedback
    let status_callback: StatusCallback =
        std::sync::Arc::new(move |status: BackendStatus| match status {
            BackendStatus::RetryAttempt {
                attempt,
                max_attempts,
                delay_ms,
                reason,
            } => {
                println!(
                    "⚠️  Retrying request (attempt {}/{}), retrying in {}ms... ({})",
                    attempt, max_attempts, delay_ms, reason
                );
            }
            BackendStatus::RateLimited {
                attempt,
                max_attempts,
                delay_ms,
            } => {
                println!(
                    "⚠️  Rate limited (attempt {}/{}), retrying in {}ms...",
                    attempt, max_attempts, delay_ms
                );
            }
            BackendStatus::NonRetryableError { message } => {
                println!("🚨 Non-retryable error detected, not retrying: {}", message);
            }
        });
    
    // Main conversation loop using the backend
    loop {
        debug!("Starting conversation turn with {} messages", messages.len());
        
        // Warn if sending many messages to cross-region models
        if uses_cross_region_model && messages.len() > 20 {
            eprintln!("⚠️  Warning: Sending {} messages to cross-region model {}.", messages.len(), selected_model);
            eprintln!("   Cross-region models have stricter limits. Consider using --new-session to start fresh.");
        }
        
        // Log token limits for cross-region models
        if uses_cross_region_model {
            debug!("Using reduced max_tokens (100) for cross-region model");
        }
        
        // Create chat request
        let chat_request = ChatRequest {
            messages: messages.clone(),
            tools: if no_tools { None } else { Some(backend_tools.clone()) },
            model: Some(selected_model.to_string()),
            inference_config: Some(InferenceConfig {
                temperature: Some(0.7),
                max_tokens: if uses_cross_region_model { Some(100) } else { Some(4096) },
                top_p: Some(0.9),
            }),
            session_id: None,
            status_callback: Some(status_callback.clone()),
        };
        
        // Send request with retry logic
        let response = backend
            .chat_with_retry(chat_request, retry_config.clone())
            .await
            .context("Failed to get response from backend")?;
        
        // Print the response text
        let content = response
            .message
            .content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text(text) => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("");
        
        if !content.is_empty() {
            print!("{}", content);
            std::io::stdout().flush().unwrap();
        }
        
        // Process tool calls if any
        if response.tool_calls.is_empty() {
            // No tool calls, save final response and exit
            if !content.is_empty() {
                let assistant_message = ContextMessage::new(MessageRole::Assistant, content);
                session_manager.add_message(&mut session, assistant_message)?;
                debug!("Saved final assistant response to session");
            }
            break;
        }
        
        // Execute tool calls
        let mut tool_results = Vec::new();
        for tool_call in &response.tool_calls {
            // Convert to JSONRPC format
            let jsonrpc_request = json!({
                "jsonrpc": "2.0",
                "method": tool_call.name,
                "params": tool_call.input,
                "id": 1
            });
            
            // Show tool execution info
            if verbose {
                println!(
                    "\n🔧 Executing tool: {} with params: {}",
                    tool_call.name,
                    serde_json::to_string_pretty(&tool_call.input)
                        .unwrap_or_else(|_| "<invalid json>".to_string())
                );
            } else {
                println!(
                    "\n🔧 Executing tool: {} with params: {}",
                    tool_call.name, tool_call.input
                );
            }
            
            debug!("Executing tool: {}", tool_call.name);
            let result = dispatcher
                .dispatch(&jsonrpc_request.to_string())
                .await
                .context("Failed to execute tool")?;
            
            // Parse result
            let parsed_result: Value =
                serde_json::from_str(&result).context("Failed to parse tool result")?;
            
            // Show results based on verbosity
            if verbose {
                println!(
                    "\n✅ Tool result for {}: {}",
                    tool_call.name,
                    serde_json::to_string_pretty(
                        parsed_result.get("result").unwrap_or(&parsed_result)
                    )
                    .unwrap_or_else(|_| "<invalid json>".to_string())
                );
            } else {
                println!("\n✅ Tool {} completed successfully", tool_call.name);
            }
            
            // Extract result content
            let result_content = if let Some(result) = parsed_result.get("result") {
                result.to_string()
            } else {
                parsed_result.to_string()
            };
            
            tool_results.push(ContentBlock::ToolResult {
                tool_call_id: tool_call.id.clone(),
                result: result_content,
            });
        }
        
        // Add assistant message with tool calls to conversation
        messages.push(BackendMessage {
            role: BackendMessageRole::Assistant,
            content: response.message.content.clone(),
        });
        
        // Add tool results as user message
        messages.push(BackendMessage {
            role: BackendMessageRole::User,
            content: tool_results,
        });
        
        // Save to session
        if !content.is_empty() {
            let assistant_message = ContextMessage::new(MessageRole::Assistant, content.clone());
            session_manager.add_message(&mut session, assistant_message)?;
        }
        
        let tool_summary = format!(
            "Tool execution results: {} tools executed",
            response.tool_calls.len()
        );
        let tool_message = ContextMessage::new(MessageRole::System, tool_summary);
        session_manager.add_message(&mut session, tool_message)?;
        
        debug!("Continuing conversation with {} messages", messages.len());
        debug!("Saved tool interaction to session");
    }
    
    // Final session save
    session_manager.save_session(&session)?;
    debug!("Final session saved: {}", session.id);
    
    // Print session info for user
    if verbose {
        println!("\n📁 Session saved: {}", session.id);
        println!("   Total messages: {}", session.messages.len());
        println!(
            "   To continue this conversation, use: --session {}",
            session.id
        );
    }
    
    Ok(())
}
