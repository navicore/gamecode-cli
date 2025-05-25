use flag_rs::{CommandBuilder, CompletionResult};
use gamecode_prompt::PromptManager;

pub fn register(parent: &mut flag_rs::Command) {
    let cmd = CommandBuilder::new("prompts")
        .short("Manage prompts")
        .build();

    parent.add_command(cmd);

    // Register subcommands
    let prompts_cmd = parent.find_subcommand_mut("prompts").unwrap();
    register_list(prompts_cmd);
    register_show(prompts_cmd);
}

fn register_list(parent: &mut flag_rs::Command) {
    let cmd = CommandBuilder::new("list")
        .short("List available prompts")
        .run(|_ctx| {
            let prompt_manager =
                PromptManager::new().map_err(|e| flag_rs::Error::Custom(Box::new(e)))?;
            let prompts = prompt_manager
                .list_prompts()
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
        .build();

    parent.add_command(cmd);
}

fn register_show(parent: &mut flag_rs::Command) {
    let cmd = CommandBuilder::new("show")
        .short("Show a specific prompt")
        .arg_completion(|_ctx, prefix| match PromptManager::new() {
            Ok(manager) => match manager.list_prompts() {
                Ok(prompts) => {
                    let mut result = CompletionResult::new();
                    for prompt_name in prompts {
                        if prompt_name.starts_with(prefix) {
                            result = result.add(prompt_name);
                        }
                    }
                    Ok(result)
                }
                Err(_) => Ok(CompletionResult::new()),
            },
            Err(_) => Ok(CompletionResult::new()),
        })
        .run(|ctx| {
            let name = ctx.args().first().ok_or_else(|| {
                flag_rs::Error::ArgumentParsing("Prompt name required".to_string())
            })?;

            let prompt_manager =
                PromptManager::new().map_err(|e| flag_rs::Error::Custom(Box::new(e)))?;
            let content = prompt_manager
                .load_prompt(name)
                .map_err(|e| flag_rs::Error::Custom(Box::new(e)))?;

            println!("{}", content);
            Ok(())
        })
        .build();

    parent.add_command(cmd);
}
