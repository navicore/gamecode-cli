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
use clap::Parser;
use gamecode_tools::{create_bedrock_dispatcher_with_schemas, schema::ToolSchemaRegistry};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::Write;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;
use tracing::debug;

#[derive(Parser, Debug)]
#[command(name = "gamecode-cli")]
#[command(about = "CLI for AWS Bedrock Claude with gamecode-tools integration")]
struct Args {
    /// The text prompt to send to Claude
    #[arg(required = true)]
    prompt: Vec<String>,

    /// The model to use (default: anthropic.claude-3-5-sonnet-20240620-v1:0)
    #[arg(long, default_value = "anthropic.claude-3-5-sonnet-20240620-v1:0")]
    model: String,

    /// AWS region
    #[arg(long, default_value = "us-east-1")]
    region: String,

    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,

    /// Maximum number of retry attempts for throttling errors
    #[arg(long, default_value = "10")]
    max_retries: usize,

    /// Initial retry delay in milliseconds
    #[arg(long, default_value = "2000")]
    initial_retry_delay_ms: u64,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

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
    debug!("Using AWS region: {}", args.region);

    let bedrock_client = Client::new(&config);

    // Setup gamecode-tools dispatcher with schema generation
    let (dispatcher, schema_registry) = create_bedrock_dispatcher_with_schemas();
    let dispatcher = Arc::new(dispatcher);

    // Combine all prompt arguments into a single string
    let user_prompt = args.prompt.join(" ");

    // Initial message to Claude
    let mut messages = vec![Message::builder()
        .role(ConversationRole::User)
        .content(ContentBlock::Text(user_prompt))
        .build()
        .context("Failed to build message")?];

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
            let model_clone = args.model.clone();
            let inference_config_clone = inference_config.clone();
            let tool_config_clone = tool_config.clone();

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

        // If no tool calls, we're done
        if tool_calls.is_empty() {
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

            debug!("Continuing conversation with {} messages", messages.len());
            // Continue the loop to let Claude respond to the tool results
            continue;
        }
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
