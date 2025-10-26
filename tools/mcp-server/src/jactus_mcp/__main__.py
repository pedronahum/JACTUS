"""Allow running MCP server as: python -m jactus_mcp"""

import asyncio
import sys

from .server import main

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nMCP server stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error starting MCP server: {e}", file=sys.stderr)
        sys.exit(1)
