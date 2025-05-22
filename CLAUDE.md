This is an experimental client for AWS Bedrock Anthropic Claude 3.7 and Haiku
3.5 models and is used as an agentic assistant.  Initial uses are programming
assistant.  The tools must all come from
https://github.com/navicore/gamecode-tools implementations.  This lib is on
github and not yet on crates.io.  This client will help us validate
gamecode-tools.  We need the cli to accept a prompt from the cli invokation args
and if the LLM responds with a tool command the cli should run the tool and
respond with the tool results and do this until the LLM has stopped sending
tool requests.  All non-tool text from the LLM should be printed to the user.
The question of what AWS Bedrock protocol and client to use is open - we want to
use an MCP-like protocol that has support for tool requests and prefer that over
the Bedrock protocal that inlines tool responses with messages.  I am not
sure what our options are and need guidance from you.
