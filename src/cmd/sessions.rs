use flag_rs::{CommandBuilder, CompletionResult};
use gamecode_context::SessionManager;
use uuid::Uuid;

pub fn register(parent: &mut flag_rs::Command) {
    let cmd = CommandBuilder::new("sessions")
        .short("Manage sessions")
        .build();
    
    parent.add_command(cmd);
    
    // Register subcommands
    let sessions_cmd = parent.find_subcommand_mut("sessions").unwrap();
    register_list(sessions_cmd);
    register_show(sessions_cmd);
    register_delete(sessions_cmd);
}

fn register_list(parent: &mut flag_rs::Command) {
    let cmd = CommandBuilder::new("list")
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
        .build();
    
    parent.add_command(cmd);
}

fn register_show(parent: &mut flag_rs::Command) {
    let cmd = CommandBuilder::new("show")
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
        .build();
    
    parent.add_command(cmd);
}

fn register_delete(parent: &mut flag_rs::Command) {
    let cmd = CommandBuilder::new("delete")
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
        .build();
    
    parent.add_command(cmd);
}