use anyhow::{Context, Result};
use aws_config::BehaviorVersion;
use aws_sdk_bedrockruntime::{
    types::{
        ContentBlock, ConversationRole, InferenceConfiguration, Message, Tool, ToolConfiguration,
        ToolInputSchema, ToolResultBlock, ToolResultContentBlock, ToolUseBlock,
    },
    Client,
};
use aws_smithy_types::{Document, Number};
use clap::{CommandFactory, Parser, Subcommand, ValueEnum, ValueHint};
use clap_complete::{generate, Generator, Shell};
use gamecode_context::{
    session::{Message as ContextMessage, MessageRole},
    SessionManager,
};
use gamecode_prompt::PromptManager;
use gamecode_tools::{create_bedrock_dispatcher_with_schemas, schema::ToolSchemaRegistry};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;
use tracing::debug;
use uuid::Uuid;

#[derive(serde::Deserialize, serde::Serialize, Clone, Debug)]
struct ModelConfig {
    pub models: HashMap<String, String>,
    pub default: String,
}

impl Default for ModelConfig {
    fn default() -> Self {
        let mut models = HashMap::new();
        models.insert(
            "claude-3.7-sonnet".to_string(),
            "us.anthropic.claude-3-7-sonnet-20250219-v1:0".to_string(),
        );
        models.insert(
            "claude-3-5-sonnet".to_string(),
            "anthropic.claude-3-5-sonnet-20240620-v1:0".to_string(),
        );
        models.insert(
            "claude-3-5-haiku".to_string(),
            "anthropic.claude-3-5-haiku-20241022-v1:0".to_string(),
        );
        models.insert(
            "claude-3-sonnet".to_string(),
            "anthropic.claude-3-sonnet-20240229-v1:0".to_string(),
        );
        models.insert(
            "claude-3-haiku".to_string(),
            "anthropic.claude-3-haiku-20240307-v1:0".to_string(),
        );

        Self {
            models,
            default: "claude-3.7-sonnet".to_string(),
        }
    }
}

impl ModelConfig {
    fn config_path() -> Result<PathBuf> {
        let home = std::env::var("HOME").context("HOME environment variable not set")?;
        Ok(PathBuf::from(home)
            .join(".config")
            .join("gamecode-cli")
            .join("models.json"))
    }

    fn load_or_create() -> Result<Self> {
        let config_path = Self::config_path()?;

        if config_path.exists() {
            let content = fs::read_to_string(&config_path).with_context(|| {
                format!("Failed to read config file: {}", config_path.display())
            })?;
            let config: ModelConfig = serde_json::from_str(&content).with_context(|| {
                format!("Failed to parse config file: {}", config_path.display())
            })?;
            Ok(config)
        } else {
            let config = Self::default();
            config.save()?;
            Ok(config)
        }
    }

    fn save(&self) -> Result<()> {
        let config_path = Self::config_path()?;

        if let Some(parent) = config_path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!("Failed to create config directory: {}", parent.display())
            })?;
        }

        let content = serde_json::to_string_pretty(self).context("Failed to serialize config")?;
        fs::write(&config_path, content)
            .with_context(|| format!("Failed to write config file: {}", config_path.display()))?;

        println!("Model config saved to: {}", config_path.display());
        Ok(())
    }

    fn get_bedrock_id(&self, model_name: &str) -> Option<&String> {
        self.models.get(model_name)
    }

    fn available_models(&self) -> Vec<String> {
        let mut models: Vec<String> = self.models.keys().cloned().collect();
        models.sort();
        models
    }
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
#[command(about = "CLI for AWS Bedrock Claude with gamecode-tools integration")]
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
    #[arg(long, value_parser = clap::builder::PossibleValuesParser::new([
        "claude-3.7-sonnet",
        "claude-3-5-sonnet",
        "claude-3-5-haiku",
        "claude-3-sonnet",
        "claude-3-haiku",
    ]))]
    model: Option<String>,

    /// AWS region
    #[arg(long, default_value = "us-west-2")]
    region: Region,

    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,

    /// Show raw JSON sent to Bedrock
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
    /// Manage model configuration
    Models {
        #[command(subcommand)]
        action: ModelAction,
    },
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
enum ModelAction {
    /// List available models
    List,
    /// Show current default model
    Default,
    /// Set default model
    SetDefault {
        /// Model name to set as default
        model: String,
    },
    /// Add a new model
    Add {
        /// Short name for the model
        name: String,
        /// Full Bedrock model ID
        bedrock_id: String,
    },
    /// Remove a model
    Remove {
        /// Model name to remove
        name: String,
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
        Some(Commands::Models { action }) => {
            let mut model_config =
                ModelConfig::load_or_create().context("Failed to load model configuration")?;

            match action {
                ModelAction::List => {
                    println!("Available models:");
                    for model in model_config.available_models() {
                        let bedrock_id = model_config.get_bedrock_id(&model).unwrap();
                        let is_default = model == model_config.default;
                        println!(
                            "  {} -> {} {}",
                            model,
                            bedrock_id,
                            if is_default { "(default)" } else { "" }
                        );
                    }
                }
                ModelAction::Default => {
                    let bedrock_id = model_config.get_bedrock_id(&model_config.default).unwrap();
                    println!("Default model: {} -> {}", model_config.default, bedrock_id);
                }
                ModelAction::SetDefault { model } => {
                    if model_config.get_bedrock_id(&model).is_none() {
                        eprintln!(
                            "Error: Unknown model '{}'. Available models: {:?}",
                            model,
                            model_config.available_models()
                        );
                        std::process::exit(1);
                    }
                    model_config.default = model.clone();
                    model_config.save()?;
                    println!("Default model set to: {}", model);
                }
                ModelAction::Add { name, bedrock_id } => {
                    model_config.models.insert(name.clone(), bedrock_id.clone());
                    model_config.save()?;
                    println!("Added model: {} -> {}", name, bedrock_id);
                }
                ModelAction::Remove { name } => {
                    if name == model_config.default {
                        eprintln!(
                            "Error: Cannot remove the default model. Set a different default first."
                        );
                        std::process::exit(1);
                    }
                    if model_config.models.remove(&name).is_some() {
                        model_config.save()?;
                        println!("Removed model: {}", name);
                    } else {
                        eprintln!("Error: Model '{}' not found", name);
                        std::process::exit(1);
                    }
                }
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
            let mut session_manager = SessionManager::new().context("Failed to create session manager")?;
            
            match action {
                SessionAction::List => {
                    match session_manager.list_sessions() {
                        Ok(sessions) => {
                            println!("Available sessions:");
                            for session_info in sessions {
                                println!("  {} (created: {:?}, messages: {})", 
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
                    }
                }
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
                                println!("  {}: {:?} - {}", i + 1, msg.role, 
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

    // Create AWS config
    let config = aws_config::defaults(BehaviorVersion::latest())
        // Use the default provider chain, which will read from environment variables
        // and the AWS config files instead of trying to directly set the region
        .load()
        .await;

    // Log which region we're using
    debug!("Using AWS region: {}", args.region.as_str());

    let bedrock_client = Client::new(&config);

    // Load model configuration
    let model_config =
        ModelConfig::load_or_create().context("Failed to load model configuration")?;

    // Determine which model to use
    let selected_model = args.model.as_deref().unwrap_or(&model_config.default);
    let bedrock_model_id = model_config
        .get_bedrock_id(selected_model)
        .with_context(|| {
            format!(
                "Unknown model '{}'. Available models: {:?}",
                selected_model,
                model_config.available_models()
            )
        })?;

    debug!("Using model: {} -> {}", selected_model, bedrock_model_id);

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
        let system_message = ContextMessage::new(MessageRole::System, system_prompt);
        session_manager.add_message(&mut session, system_message)?;
    }

    // Add current user prompt to session
    let user_prompt = args.prompt.join(" ");
    let user_message = ContextMessage::new(MessageRole::User, user_prompt);
    session_manager.add_message(&mut session, user_message)?;

    // Convert session messages to Bedrock format
    let mut messages = Vec::new();
    for context_msg in &session.messages {
        let role = match context_msg.role {
            MessageRole::System => ConversationRole::User, // Bedrock treats system as user
            MessageRole::User => ConversationRole::User,
            MessageRole::Assistant => ConversationRole::Assistant,
            MessageRole::Tool => ConversationRole::User, // Tool messages treated as user context
        };

        let message = Message::builder()
            .role(role)
            .content(ContentBlock::Text(context_msg.content.clone()))
            .build()
            .context("Failed to build message from session")?;

        messages.push(message);
    }

    // Define tool specifications using dynamic schemas
    let tools = get_dynamic_tool_specs(&schema_registry)?;

    // Create tool configuration
    // Create tool configuration with all tools
    let mut tool_config_builder = ToolConfiguration::builder();

    // Add each tool individually
    for tool in tools {
        tool_config_builder = tool_config_builder.tools(tool);
    }

    let tool_config = tool_config_builder
        .build()
        .context("Failed to build tool configuration")?;

    // Set inference configuration
    let inference_config = InferenceConfiguration::builder()
        .temperature(0.7)
        .top_p(0.9)
        .max_tokens(4096)
        .build();

    // Main conversation loop
    loop {
        debug!(
            "Starting conversation turn with {} messages",
            messages.len()
        );
        // Send request to Bedrock with retry logic
        let mut response = {
            let messages_clone = messages.clone();
            let bedrock_client_clone = bedrock_client.clone();
            let model_clone = bedrock_model_id.clone();
            let inference_config_clone = inference_config.clone();
            let tool_config_clone = tool_config.clone();

            let debug_flag = args.debug;
            retry_with_backoff(
                || async {
                    // Create request builder
                    let mut request_builder = bedrock_client_clone
                        .converse_stream()
                        .model_id(&model_clone)
                        .inference_config(inference_config_clone.clone())
                        .tool_config(tool_config_clone.clone());

                    // Add each message individually
                    for message in &messages_clone {
                        request_builder = request_builder.messages(message.clone());
                    }

                    // Debug: Show the request information
                    if debug_flag {
                        println!("\n🔍 DEBUG: Bedrock Request Details:");
                        println!("Model ID: {}", &model_clone);
                        println!("Messages ({}):", messages_clone.len());

                        for (i, msg) in messages_clone.iter().enumerate() {
                            println!("  Message {}: Role = {:?}", i + 1, msg.role());
                            for (j, content) in msg.content().iter().enumerate() {
                                match content {
                                    ContentBlock::Text(text) => {
                                        let preview = if text.len() > 100 {
                                            format!("{}...", &text[..100])
                                        } else {
                                            text.clone()
                                        };
                                        println!("    Content {}: Text = \"{}\"", j + 1, preview);
                                    },
                                    ContentBlock::ToolUse(tool_use) => {
                                        println!("    Content {}: ToolUse = {{ name: \"{}\", id: \"{}\" }}",
                                               j + 1, tool_use.name(), tool_use.tool_use_id());
                                    },
                                    ContentBlock::ToolResult(tool_result) => {
                                        println!("    Content {}: ToolResult = {{ id: \"{}\" }}",
                                               j + 1, tool_result.tool_use_id());
                                    },
                                    _ => println!("    Content {}: Other", j + 1)
                                }
                            }
                        }

                        println!("Inference Config:");
                        println!("  Temperature: {:?}", inference_config_clone.temperature());
                        println!("  Top P: {:?}", inference_config_clone.top_p());
                        println!("  Max Tokens: {:?}", inference_config_clone.max_tokens());

                        let tools = tool_config_clone.tools();
                        if !tools.is_empty() {
                            println!("Tools ({}):", tools.len());
                            for (i, tool) in tools.iter().enumerate() {
                                match tool.as_tool_spec() {
                                    Ok(spec) => {
                                        println!("  Tool {}: {} - {}",
                                               i + 1,
                                               spec.name(),
                                               spec.description().unwrap_or("no description"));
                                    },
                                    Err(_) => {
                                        println!("  Tool {}: Unknown tool type", i + 1);
                                    }
                                }
                            }
                        }
                        println!();
                    }

                    // Send the request
                    request_builder.send().await
                },
                args.max_retries,
                Duration::from_millis(args.initial_retry_delay_ms),
                args.verbose,
            )
            .await
            .context("Failed to send request to Bedrock after retries")?
        };

        // Process streaming response
        let mut current_tool_use: Option<(String, String, String)> = None; // (name, id, input)
        let mut response_text = String::new();
        #[allow(unused_variables)]
        let mut role = String::new();
        let mut tool_calls = Vec::new();
        let mut tool_use_blocks = Vec::new();

        // Process each event in the stream using traditional async methods
        loop {
            // Receive next event or break if done
            let result = response.stream.recv().await;

            // Handle outer Result first (SDK error)
            match result {
                Ok(Some(event)) => {
                    // Process valid event
                    // Handle different event types
                    #[allow(unused_assignments)]
                    if let Ok(message_start) = event.as_message_start() {
                        role = message_start.role().as_str().to_string();
                    }

                    if let Ok(content_start) = event.as_content_block_start() {
                        if let Some(start) = content_start.start() {
                            if let Ok(tool_use) = start.as_tool_use() {
                                // Start of tool use - capture name and ID
                                current_tool_use = Some((
                                    tool_use.name().to_string(),
                                    tool_use.tool_use_id().to_string(),
                                    String::new(),
                                ));
                            }
                        }
                    }

                    if let Ok(content_delta) = event.as_content_block_delta() {
                        if let Some(delta) = content_delta.delta() {
                            if let Ok(text) = delta.as_text() {
                                // Regular text response
                                response_text.push_str(text);
                                print!("{}", text);
                                std::io::stdout().flush().unwrap();
                            } else if let Ok(tool_use) = delta.as_tool_use() {
                                // Tool use input accumulation
                                if let Some(ref mut current) = current_tool_use {
                                    current.2.push_str(tool_use.input());
                                }
                            }
                        }
                    }

                    // Handle tool use completion
                    if let Ok(_content_stop) = event.as_content_block_stop() {
                        if let Some((name, id, input)) = current_tool_use.take() {
                            // Parse the accumulated input
                            let tool_input = match serde_json::from_str(&input) {
                                Ok(parsed) => parsed,
                                Err(_) => json!({"param": "value"}),
                            };

                            // Create tool use block for inclusion in assistant message
                            let tool_use_block = ToolUseBlock::builder()
                                .tool_use_id(&id)
                                .name(&name)
                                .input(json_value_to_document(&tool_input))
                                .build()
                                .context("Failed to build tool use block")?;

                            tool_use_blocks.push(ContentBlock::ToolUse(tool_use_block));
                            tool_calls.push((name, id, tool_input));
                        }
                    }
                }
                Ok(None) => {
                    // End of stream
                    break;
                }
                Err(err) => {
                    eprintln!("\nStream error: {}", err);
                    break;
                }
            }
        }

        // If no tool calls, save final assistant response and we're done
        if tool_calls.is_empty() {
            if !response_text.is_empty() {
                // Save the assistant's final response to the session
                let assistant_message = ContextMessage::new(MessageRole::Assistant, response_text);
                session_manager.add_message(&mut session, assistant_message)?;
                debug!("Saved final assistant response to session");
            }
            break;
        }

        // Process tool calls and append results
        let mut tool_results = Vec::new();

        for (tool_name, tool_id, tool_args) in &tool_calls {
            // Convert Bedrock tool args format to JSONRPC format
            let jsonrpc_request = json!({
                "jsonrpc": "2.0",
                "method": tool_name,
                "params": tool_args,
                "id": 1
            });

            // Show tool execution info
            if args.verbose {
                println!(
                    "\n🔧 Executing tool: {} with params: {}",
                    tool_name,
                    serde_json::to_string_pretty(tool_args)
                        .unwrap_or_else(|_| "<invalid json>".to_string())
                );
            } else {
                println!(
                    "\n🔧 Executing tool: {} with params: {}",
                    tool_name, tool_args
                );
            }

            debug!("Executing tool: {}", tool_name);
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
                    tool_name,
                    serde_json::to_string_pretty(
                        parsed_result.get("result").unwrap_or(&parsed_result)
                    )
                    .unwrap_or_else(|_| "<invalid json>".to_string())
                );
            } else {
                println!("\n✅ Tool {} completed successfully", tool_name);
            }

            // Create tool result block
            let result_json = if let Some(result) = parsed_result.get("result") {
                result.to_string()
            } else {
                // If no result field, use the entire response
                parsed_result.to_string()
            };

            // Create tool result with proper format
            let tool_result = ToolResultBlock::builder()
                .tool_use_id(tool_id.clone())
                .content(ToolResultContentBlock::Text(result_json))
                .build()
                .context("Failed to build tool result")?;

            tool_results.push(tool_result);
        }

        // Continue the conversation properly by adding assistant message with tool use
        // and user message with tool results
        if !tool_results.is_empty() {
            // First, add the assistant's message with the tool use blocks
            let mut assistant_content = vec![];

            // Add any text response from the assistant
            if !response_text.is_empty() {
                assistant_content.push(ContentBlock::Text(response_text.clone()));
            }

            // Add all tool use blocks
            assistant_content.extend(tool_use_blocks);

            let assistant_message = Message::builder()
                .role(ConversationRole::Assistant)
                .set_content(Some(assistant_content))
                .build()
                .context("Failed to build assistant message")?;

            messages.push(assistant_message);

            // Then add a user message with the tool results
            let mut user_content = vec![];

            for tool_result in tool_results {
                user_content.push(ContentBlock::ToolResult(tool_result));
            }

            let user_message = Message::builder()
                .role(ConversationRole::User)
                .set_content(Some(user_content))
                .build()
                .context("Failed to build user message with tool results")?;

            messages.push(user_message);

            // Save the assistant's response with tool calls to the session
            if !response_text.is_empty() {
                let assistant_message =
                    ContextMessage::new(MessageRole::Assistant, response_text.clone());
                session_manager.add_message(&mut session, assistant_message)?;
            }

            // Save tool results as a system message (for context but not displayed)
            let tool_summary = format!(
                "Tool execution results: {} tools executed",
                tool_calls.len()
            );
            let tool_message = ContextMessage::new(MessageRole::System, tool_summary);
            session_manager.add_message(&mut session, tool_message)?;

            debug!("Continuing conversation with {} messages", messages.len());
            debug!("Saved tool interaction to session");
            // Continue the loop to let Claude respond to the tool results
            continue;
        }
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

// Helper function to extract concise error information
fn extract_error_summary(error_str: &str) -> String {
    // Look for common AWS error patterns
    if let Some(start) = error_str.find("ServiceError") {
        if let Some(end) = error_str[start..].find(", ") {
            let service_error = &error_str[start..start + end];
            // Extract just the error type from ServiceError { err: ErrorType, ... }
            if let Some(err_start) = service_error.find("err: ") {
                if let Some(err_end) = service_error[err_start..].find("(") {
                    let error_type = &service_error[err_start + 5..err_start + err_end];
                    return format!("ServiceError source: {}", error_type);
                }
            }
            return service_error.to_string();
        }
    }

    // Look for ThrottlingException directly
    if error_str.contains("ThrottlingException") {
        return "ThrottlingException".to_string();
    }

    // Look for ValidationException
    if error_str.contains("ValidationException") {
        return "ValidationException".to_string();
    }

    // Look for other common patterns
    if error_str.contains("Too many requests") {
        return "Rate limit exceeded".to_string();
    }

    // If we can't parse it, just take the first part
    if let Some(newline_pos) = error_str.find('\n') {
        error_str[..newline_pos].to_string()
    } else if error_str.len() > 100 {
        format!("{}...", &error_str[..100])
    } else {
        error_str.to_string()
    }
}

// Helper function to perform exponential backoff retry for AWS Bedrock calls
async fn retry_with_backoff<F, Fut, T, E>(
    operation: F,
    max_retries: usize,
    initial_delay: Duration,
    verbose: bool,
) -> Result<T>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = std::result::Result<T, E>>,
    E: std::fmt::Display + std::fmt::Debug,
{
    let mut delay = initial_delay;
    let mut last_error = None;

    for attempt in 0..=max_retries {
        match operation().await {
            Ok(result) => return Ok(result),
            Err(error) => {
                let error_str = format!("{:?}", error);

                // Extract concise error info for non-verbose mode
                let concise_error = extract_error_summary(&error_str);

                if verbose {
                    eprintln!(
                        "\n🚨 Bedrock request failed (attempt {}/{}): {}",
                        attempt + 1,
                        max_retries + 1,
                        error_str
                    );
                } else {
                    eprintln!(
                        "\n🚨 Bedrock request failed (attempt {}/{}): {}",
                        attempt + 1,
                        max_retries + 1,
                        concise_error
                    );
                }

                // Check if this is a throttling error
                let is_throttling = error_str.contains("ThrottlingException")
                    || error_str.contains("Too many requests")
                    || error_str.contains("Rate exceeded")
                    || error_str.contains("Throttled");

                // Check if this is a validation error (don't retry these)
                let is_validation_error = error_str.contains("ValidationException")
                    || error_str.contains("InvalidRequest")
                    || error_str.contains("BadRequest")
                    || error_str.contains("content too large")
                    || error_str.contains("exceeds maximum")
                    || error_str.contains("invalid");

                if attempt == max_retries {
                    // Last attempt - don't retry
                    last_error = Some(error);
                    break;
                }

                if is_validation_error {
                    // Don't retry validation errors
                    if verbose {
                        eprintln!("⚠️  Validation error detected, not retrying: {}", error_str);
                    } else {
                        eprintln!(
                            "⚠️  Validation error detected, not retrying: {}",
                            concise_error
                        );
                    }
                    last_error = Some(error);
                    break;
                }

                if is_throttling {
                    println!(
                        "⚠️  Rate limited by AWS Bedrock (attempt {}/{}), retrying in {}ms...",
                        attempt + 1,
                        max_retries + 1,
                        delay.as_millis()
                    );
                } else {
                    println!(
                        "⚠️  Retrying Bedrock request (attempt {}/{}), retrying in {}ms...",
                        attempt + 1,
                        max_retries + 1,
                        delay.as_millis()
                    );
                }

                if is_throttling {
                    debug!(
                        "Attempt {}/{} failed with throttling error, retrying after {}ms: {}",
                        attempt + 1,
                        max_retries + 1,
                        delay.as_millis(),
                        error_str
                    );
                } else {
                    debug!(
                        "Attempt {}/{} failed, retrying after {}ms: {}",
                        attempt + 1,
                        max_retries + 1,
                        delay.as_millis(),
                        error_str
                    );
                }

                sleep(delay).await;

                // Exponential backoff with longer delays (up to 3x)
                delay = std::cmp::min(
                    delay * 3,
                    Duration::from_secs(60), // Cap at 60 seconds
                );

                last_error = Some(error);
            }
        }
    }

    // If we get here, all retries failed
    if let Some(error) = last_error {
        return Err(anyhow::anyhow!(
            "All retry attempts failed. Last error: {}",
            error
        ));
    }

    unreachable!("Should not reach here")
}

// Helper function to convert JSON Value to Document
fn json_value_to_document(value: &Value) -> Document {
    match value {
        Value::String(s) => Document::String(s.clone()),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Document::Number(Number::PosInt(i as u64))
            } else if let Some(f) = n.as_f64() {
                Document::Number(Number::Float(f))
            } else {
                Document::String(n.to_string())
            }
        }
        Value::Bool(b) => Document::Bool(*b),
        Value::Null => Document::Null,
        Value::Array(arr) => {
            let doc_vec: Vec<Document> = arr.iter().map(json_value_to_document).collect();
            Document::Array(doc_vec)
        }
        Value::Object(obj) => {
            let mut doc_map = HashMap::new();
            for (k, v) in obj {
                doc_map.insert(k.clone(), json_value_to_document(v));
            }
            Document::Object(doc_map)
        }
    }
}

// Helper function to create a JSON schema Document
#[allow(dead_code)]
fn create_schema_document(properties: Vec<(&str, &str, &str)>, required: Vec<&str>) -> Document {
    let mut props = HashMap::new();
    for (name, type_str, desc) in properties {
        props.insert(
            name.to_string(),
            Document::Object(HashMap::from([
                ("type".into(), Document::String(type_str.into())),
                ("description".into(), Document::String(desc.into())),
            ])),
        );
    }

    Document::Object(HashMap::from([
        ("type".into(), Document::String("object".into())),
        ("properties".into(), Document::Object(props)),
        (
            "required".into(),
            Document::Array(
                required
                    .into_iter()
                    .map(|r| Document::String(r.into()))
                    .collect(),
            ),
        ),
    ]))
}

// Helper function to define tool specifications using dynamic schemas
fn get_dynamic_tool_specs(schema_registry: &ToolSchemaRegistry) -> Result<Vec<Tool>> {
    let mut tools = Vec::new();

    // Convert each tool schema to Bedrock format
    for bedrock_spec in schema_registry.to_bedrock_specs() {
        let tool_spec = aws_sdk_bedrockruntime::types::ToolSpecification::builder()
            .name(&bedrock_spec.name)
            .description(&bedrock_spec.description)
            .input_schema(ToolInputSchema::Json(json_value_to_document(
                &bedrock_spec.input_schema.json,
            )))
            .build()
            .context("Failed to build tool specification")?;

        tools.push(Tool::ToolSpec(tool_spec));
    }

    Ok(tools)
}

// Legacy function to define tool specifications (kept for reference)
#[allow(dead_code)]
fn get_tool_specs() -> Result<Vec<Tool>> {
    let tool_specs = vec![
        // File operations
        Tool::ToolSpec(
            aws_sdk_bedrockruntime::types::ToolSpecification::builder()
                .name("file_read")
                .description("Read a file from the filesystem")
                .input_schema(ToolInputSchema::Json(create_schema_document(
                    vec![("path", "string", "Path to the file to read")],
                    vec!["path"],
                )))
                .build()
                .context("Failed to build file_read tool spec")?,
        ),
        Tool::ToolSpec(
            aws_sdk_bedrockruntime::types::ToolSpecification::builder()
                .name("file_write")
                .description("Write content to a file")
                .input_schema(ToolInputSchema::Json(create_schema_document(
                    vec![
                        ("path", "string", "Path to the file to write"),
                        ("content", "string", "Content to write to the file"),
                    ],
                    vec!["path", "content"],
                )))
                .build()
                .context("Failed to build file_write tool spec")?,
        ),
        Tool::ToolSpec(
            aws_sdk_bedrockruntime::types::ToolSpecification::builder()
                .name("file_patch")
                .description("Apply a patch to a file")
                .input_schema(ToolInputSchema::Json(create_schema_document(
                    vec![
                        ("path", "string", "Path to the file to patch"),
                        ("original", "string", "Original content to replace"),
                        ("modified", "string", "New content to insert"),
                    ],
                    vec!["path", "original", "modified"],
                )))
                .build()
                .context("Failed to build file_patch tool spec")?,
        ),
        // Directory operations
        Tool::ToolSpec(
            aws_sdk_bedrockruntime::types::ToolSpecification::builder()
                .name("directory_list")
                .description("List contents of a directory")
                .input_schema(ToolInputSchema::Json(create_schema_document(
                    vec![("path", "string", "Directory path to list")],
                    vec!["path"],
                )))
                .build()
                .context("Failed to build directory_list tool spec")?,
        ),
        Tool::ToolSpec(
            aws_sdk_bedrockruntime::types::ToolSpecification::builder()
                .name("directory_make")
                .description("Create a directory")
                .input_schema(ToolInputSchema::Json(create_schema_document(
                    vec![("path", "string", "Directory path to create")],
                    vec!["path"],
                )))
                .build()
                .context("Failed to build directory_make tool spec")?,
        ),
        // Search operations
        Tool::ToolSpec(
            aws_sdk_bedrockruntime::types::ToolSpecification::builder()
                .name("file_find")
                .description("Find files matching a pattern")
                .input_schema(ToolInputSchema::Json(create_schema_document(
                    vec![
                        ("path", "string", "Base directory for search"),
                        ("pattern", "string", "Glob pattern to match files"),
                    ],
                    vec!["path", "pattern"],
                )))
                .build()
                .context("Failed to build file_find tool spec")?,
        ),
        Tool::ToolSpec(
            aws_sdk_bedrockruntime::types::ToolSpecification::builder()
                .name("file_grep")
                .description("Search file contents for a pattern")
                .input_schema(ToolInputSchema::Json(create_schema_document(
                    vec![
                        ("path", "string", "File or directory to search"),
                        ("pattern", "string", "Regular expression to search for"),
                    ],
                    vec!["path", "pattern"],
                )))
                .build()
                .context("Failed to build file_grep tool spec")?,
        ),
        // Shell execution
        Tool::ToolSpec(
            aws_sdk_bedrockruntime::types::ToolSpecification::builder()
                .name("shell")
                .description("Execute a shell command")
                .input_schema(ToolInputSchema::Json(create_schema_document(
                    vec![("command", "string", "Command to execute")],
                    vec!["command"],
                )))
                .build()
                .context("Failed to build shell tool spec")?,
        ),
    ];

    Ok(tool_specs)
}
