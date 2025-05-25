use flag_rs::{CommandBuilder, CompletionResult, Shell};

pub fn register(parent: &mut flag_rs::Command) {
    let cmd = CommandBuilder::new("completion")
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
                    result = result.add_with_description(shell, desc);
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
            
            // We need to rebuild the root command to generate completion
            let root = crate::build_cli();
            let completion = root.generate_completion(shell);
            // Fix flag-rs bug: replace hyphens with underscores in shell identifiers
            let fixed_completion = completion
                .replace("GAMECODE-CLI_COMPLETE", "GAMECODE_CLI_COMPLETE")
                .replace("_gamecode-cli_complete", "_gamecode_cli_complete");
            println!("{}", fixed_completion);
            
            Ok(())
        })
        .build();

    parent.add_command(cmd);
}