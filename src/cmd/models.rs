use flag_rs::CommandBuilder;

pub fn register(parent: &mut flag_rs::Command) {
    let cmd = CommandBuilder::new("models")
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
        .build();

    parent.add_command(cmd);
}