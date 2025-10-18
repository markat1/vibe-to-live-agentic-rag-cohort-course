from fastmcp import FastMCP

mcp = FastMCP("weather-server")
mcp.run(transport="streamable-http", host="0.0.0.0", port=8000)

@mcp.tool
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@mcp.resource("config://version")
def get_version():
    return "2.0.1"

@mcp.resource("users://{user_id}/profile")
def get_profile(user_id: int):
    return {"name": f"User {user_id}", "status": "active"}

if __name__ == "__main__":
    mcp.run() # Runs the server in STDIO mode by default