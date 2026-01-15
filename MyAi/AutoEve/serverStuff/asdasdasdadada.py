import socket
import threading

HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
PORT = 65432        # Port to listen on (non-privileged ports are > 1023)

clients = []

def handle_client(conn, addr):
    print('Connected by', addr)
    clients.append(conn)
    try:
        while True:
            data = conn.recv(1024)
            if not data:
                break
            print(f"Received from {addr}: {data.decode()}")
            for client in clients:
                if client != conn:
                    client.sendall(data)
            print(f"Sent to all clients: {data.decode()}")
    except ConnectionResetError:
        print(f"Connection with {addr} reset")
    finally:
        clients.remove(conn)
        conn.close()
        print(f"Connection with {addr} closed")


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    print(f"Server listening on {HOST}:{PORT}")
    while True:
        conn, addr = s.accept()
        client_thread = threading.Thread(target=handle_client, args=(conn, addr))
        client_thread.start()