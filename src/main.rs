use anyhow::{Context, Result};
use clap::{CommandFactory, Parser, Subcommand, ValueEnum, ValueHint};
use clap_complete::{generate, Generator, Shell};
use gamecode_backend::{
    ChatRequest, ContentBlock, InferenceConfig, LLMBackend, Message as BackendMessage,
    MessageRole as BackendMessageRole, RetryConfig, Tool as BackendTool,
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

#[derive(ValueEnum, Clone, Debug)]
enum Region {
    #[value(name = "us-east-1")]
    UsEast1,
    #[value(name = "us-east-2")]
    UsEast2,
    #[value(name = "us-west-1")]
    UsWest1,
    #[value(name = "us-west-2")]
    UsWest2,
    #[value(name = "eu-west-1")]
    EuWest1,
    #[value(name = "eu-west-2")]
    EuWest2,
    #[value(name = "eu-central-1")]
    EuCentral1,
    #[value(name = "ap-southeast-1")]
    ApSoutheast1,
    #[value(name = "ap-southeast-2")]
    ApSoutheast2,
    #[value(name = "ap-northeast-1")]
    ApNortheast1,
}

impl Region {
    fn as_str(&self) -> &'static str {
        match self {
            Region::UsEast1 => "us-east-1",
            Region::UsEast2 => "us-east-2",
            Region::UsWest1 => "us-west-1",
            Region::UsWest2 => "us-west-2",
            Region::EuWest1 => "eu-west-1",
            Region::EuWest2 => "eu-west-2",
            Region::EuCentral1 => "eu-central-1",
            Region::ApSoutheast1 => "ap-southeast-1",
            Region::ApSoutheast2 => "ap-southeast-2",
            Region::ApNortheast1 => "ap-northeast-1",
        }
    }
}

#[derive(Parser, Debug)]
#[command(name = "gamecode-cli")]
#[command(about = "CLI for Claude AI with gamecode-tools integration")]
struct Args {
    #[command(subcommand)]
    command: Option<Commands>,

    /// The text prompt to send to Claude
    #[arg(long, short = 'p', value_hint = ValueHint::Other)]
    prompt: Vec<String>,

    /// Named prompt to use for system prompt (uses default if not specified)
    #[arg(long, value_parser = clap::builder::PossibleValuesParser::new([
        "default",
        "coding",
        "code-review",
        "debugging",
    ]))]
    system_prompt: Option<String>,

    /// The model to use
    #[arg(long)]
    model: Option<String>,

    /// AWS region
    #[arg(long, default_value = "us-west-2")]
    region: Region,

    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,

    /// Show debug information
    #[arg(short, long)]
    debug: bool,

    /// Session ID to continue an existing conversation (use 'gamecode-cli sessions list' to see available sessions)
    #[arg(long, value_hint = ValueHint::Other)]
    session: Option<String>,

    /// Start a new session (ignore any existing session)
    #[arg(long)]
    new_session: bool,

    /// Maximum number of retry attempts for throttling errors
    #[arg(long, default_value = "10")]
    max_retries: usize,

    /// Initial retry delay in milliseconds
    #[arg(long, default_value = "2000")]
    initial_retry_delay_ms: u64,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Generate shell completions
    #[command(arg_required_else_help = true)]
    Completions {
        /// The shell to generate completions for
        #[arg(value_enum)]
        shell: Shell,
    },
    /// List available models
    Models,
    /// Manage prompts
    Prompts {
        #[command(subcommand)]
        action: PromptAction,
    },
    /// Manage sessions
    Sessions {
        #[command(subcommand)]
        action: SessionAction,
    },
}

#[derive(Subcommand, Debug)]
enum PromptAction {
    /// List available prompts
    List,
    /// Show a specific prompt
    Show {
        /// Prompt name to show
        name: String,
    },
}

#[derive(Subcommand, Debug)]
enum SessionAction {
    /// List available sessions
    List,
    /// Show session details
    Show {
        /// Session ID to show
        id: String,
    },
    /// Delete a session
    Delete {
        /// Session ID to delete
        id: String,
    },
}

fn print_completions<G: Generator>(generator: G, cmd: &mut clap::Command) {
    generate(
        generator,
        cmd,
        cmd.get_name().to_string(),
        &mut std::io::stdout(),
    );
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
    let args = Args::parse();

    // Handle subcommands
    match args.command {
        Some(Commands::Completions { shell }) => {
            let mut cmd = Args::command();
            print_completions(shell, &mut cmd);
            return Ok(());
        }
        Some(Commands::Models) => {
            // Create a backend to query supported models
            let backend = create_backend(args.region.as_str()).await?;
            let supported_models = backend.supported_models();

            println!("Supported models:");
            for model in supported_models {
                println!("  {}", model);
            }
            return Ok(());
        }
        Some(Commands::Prompts { action }) => {
            let prompt_manager = PromptManager::new().context("Failed to create prompt manager")?;

            match action {
                PromptAction::List => match prompt_manager.list_prompts() {
                    Ok(prompts) => {
                        println!("Available prompts:");
                        for prompt in prompts {
                            println!("  {}", prompt);
                        }
                    }
                    Err(e) => {
                        eprintln!("Error listing prompts: {}", e);
                        std::process::exit(1);
                    }
                },
                PromptAction::Show { name } => match prompt_manager.load_prompt(&name) {
                    Ok(content) => {
                        println!("Prompt '{}': {}", name, content);
                    }
                    Err(e) => {
                        eprintln!("Error reading prompt: {}", e);
                        std::process::exit(1);
                    }
                },
            }
            return Ok(());
        }
        Some(Commands::Sessions { action }) => {
            let mut session_manager =
                SessionManager::new().context("Failed to create session manager")?;

            match action {
                SessionAction::List => match session_manager.list_sessions() {
                    Ok(sessions) => {
                        println!("Available sessions:");
                        for session_info in sessions {
                            println!(
                                "  {} (created: {:?}, messages: {})",
                                session_info.id,
                                session_info.created_at,
                                session_info.message_count
                            );
                        }
                    }
                    Err(e) => {
                        eprintln!("Error listing sessions: {}", e);
                        std::process::exit(1);
                    }
                },
                SessionAction::Show { id } => {
                    let session_id = Uuid::parse_str(&id)
                        .with_context(|| format!("Invalid session ID: {}", id))?;
                    match session_manager.load_session(&session_id) {
                        Ok(session) => {
                            println!("Session: {}", session.id);
                            println!("Created: {:?}", session.created_at);
                            println!("Updated: {:?}", session.updated_at);
                            println!("Messages: {}", session.messages.len());
                            for (i, msg) in session.messages.iter().enumerate() {
                                println!(
                                    "  {}: {:?} - {}",
                                    i + 1,
                                    msg.role,
                                    if msg.content.len() > 100 {
                                        format!("{}...", &msg.content[..100])
                                    } else {
                                        msg.content.clone()
                                    }
                                );
                            }
                        }
                        Err(e) => {
                            eprintln!("Error loading session: {}", e);
                            std::process::exit(1);
                        }
                    }
                }
                SessionAction::Delete { id } => {
                    // For now, just show that the feature would delete the session
                    // since the SessionManager might not have a delete method yet
                    println!("Session deletion feature not yet implemented for {}", id);
                    println!("Sessions are stored as files and can be manually deleted if needed");
                }
            }
            return Ok(());
        }
        None => {}
    }

    // Require prompt when not using subcommands
    if args.prompt.is_empty() {
        eprintln!("Error: prompt is required when not using a subcommand");
        eprintln!("Use --prompt \"your message here\" or -p \"your message here\"");
        std::process::exit(1);
    }

    // Setup logging
    let log_level = if args.verbose {
        tracing::Level::DEBUG
    } else {
        tracing::Level::INFO
    };
    tracing_subscriber::fmt().with_max_level(log_level).init();

    // Create backend with region and model configuration
    debug!("Using AWS region: {}", args.region.as_str());

    let backend = create_backend(args.region.as_str()).await?;

    // Use default model if none specified
    let selected_model = args.model.as_deref().unwrap_or("claude-3.7-sonnet");
    debug!("Using model: {}", selected_model);

    // Setup gamecode-tools dispatcher with schema generation
    let (dispatcher, schema_registry) = create_bedrock_dispatcher_with_schemas();
    let dispatcher = Arc::new(dispatcher);

    // Setup session management
    let mut session_manager = SessionManager::new().context("Failed to create session manager")?;

    // Load or create session based on arguments
    let mut session = if args.new_session {
        debug!("Creating new session");
        session_manager.new_session()?
    } else if let Some(session_id_str) = &args.session {
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
        let prompt_manager = PromptManager::new().context("Failed to create prompt manager")?;
        let system_prompt = if let Some(prompt_name) = &args.system_prompt {
            prompt_manager
                .load_prompt(prompt_name)
                .with_context(|| format!("Failed to load prompt '{}'", prompt_name))?
        } else {
            prompt_manager
                .load_default()
                .context("Failed to load default prompt")?
        };

        if args.verbose {
            if let Some(prompt_name) = &args.system_prompt {
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
    let user_prompt = args.prompt.join(" ");
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
    let backend_tools = convert_tools_to_backend(&schema_registry)?;

    // Create retry configuration
    let retry_config = RetryConfig {
        max_retries: args.max_retries,
        initial_delay: Duration::from_millis(args.initial_retry_delay_ms),
        backoff_strategy: gamecode_backend::BackoffStrategy::Exponential { multiplier: 3 },
        verbose: args.verbose,
    };

    // Main conversation loop using the backend
    loop {
        debug!(
            "Starting conversation turn with {} messages",
            messages.len()
        );

        // Create chat request
        let chat_request = ChatRequest {
            messages: messages.clone(),
            tools: Some(backend_tools.clone()),
            model: args.model.clone(),
            inference_config: Some(InferenceConfig {
                temperature: Some(0.7),
                max_tokens: Some(4096),
                top_p: Some(0.9),
            }),
            session_id: None,
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
            if args.verbose {
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
            if args.verbose {
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
    if args.verbose {
        println!("\n📁 Session saved: {}", session.id);
        println!("   Total messages: {}", session.messages.len());
        println!(
            "   To continue this conversation, use: --session {}",
            session.id
        );
    }

    Ok(())
}
