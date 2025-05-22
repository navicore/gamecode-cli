# Retry Configuration Examples

The gamecode-cli supports automatic retry with exponential backoff for handling AWS Bedrock throttling errors.

## Basic Usage

```bash
# Default retry settings (3 retries, 1000ms initial delay)
./gamecode-cli "list files in current directory"
```

## Custom Retry Settings

```bash
# More aggressive retry (5 attempts, start with 500ms delay)  
./gamecode-cli --max-retries 5 --initial-retry-delay-ms 500 "read the README.md file"

# Conservative retry (1 attempt, 2000ms delay)
./gamecode-cli --max-retries 1 --initial-retry-delay-ms 2000 "help me analyze the source code"

# No retries (fail fast)
./gamecode-cli --max-retries 0 "create a new file called test.txt"
```

## How Retry Works

1. **Exponential Backoff**: Each retry doubles the delay (500ms → 1000ms → 2000ms → ...)
2. **Maximum Delay**: Capped at 30 seconds to prevent excessive waiting
3. **Throttling Detection**: Automatically detects AWS throttling errors:
   - `ThrottlingException`
   - `Too many requests`
   - `Rate exceeded`
   - `Throttled`
4. **User Feedback**: Shows friendly messages during retries:
   ```
   ⚠️  Rate limited by AWS Bedrock (attempt 1/3), retrying in 1000ms...
   ```

## Best Practices

- **Development**: Use default settings (3 retries, 1000ms)
- **Production Scripts**: Consider higher max-retries (5-10) with longer initial delays
- **Interactive Use**: Default settings provide good balance of responsiveness and reliability
- **Batch Processing**: Use longer delays to be more respectful of API limits

## Verbose Logging

Enable verbose logging to see detailed retry information:

```bash
./gamecode-cli --verbose --max-retries 3 "your command here"
```