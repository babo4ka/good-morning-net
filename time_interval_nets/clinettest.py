import socket
import json
import math

HOST = ('localhost', 9999)

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(HOST)

client.send(b'e')

res = client.recv(4096)
res = json.loads(res.decode())

start = res.get("start")
end = res.get("end")

start_h, start_m = math.floor(start), start - math.floor(start)
end_h, end_m = math.floor(end), end - math.floor(end)

print(res)
print(start_h, int(start_m * 60))
print(end_h, int(end_m * 60))