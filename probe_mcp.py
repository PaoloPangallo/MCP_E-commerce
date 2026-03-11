import urllib.request
import urllib.error

urls = [
    'http://127.0.0.1:8050/mcp',
    'http://127.0.0.1:8050/mcp/sse',
    'http://127.0.0.1:8050/mcp/message',
    'http://127.0.0.1:8050/mcp/mcp',
    'http://127.0.0.1:8050/mcp/mcp/sse',
    'http://127.0.0.1:8050/mcp/mcp/message',
]

for u in urls:
    try:
        req = urllib.request.Request(u, headers={'Accept': 'text/event-stream'})
        res = urllib.request.urlopen(req)
        print(f'{u}: {res.status}')
    except urllib.error.HTTPError as e:
        print(f'{u}: HTTP {e.code}')
    except Exception as e:
        print(f'{u}: {e}')
