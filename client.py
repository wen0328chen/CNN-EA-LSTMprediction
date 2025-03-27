import socket


def main():
    host = '127.0.0.1'  # 服务器IP地址，根据实际情况修改
    port = 12345  # 服务器端口号，与服务器保持一致

    # 建立 TCP 连接
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))
    print("Connected to server {}:{}".format(host, port))

    # 输入初始请求（例如：“我想要预测1小时后的空气质量”）
    request = input("Please enter the request content:")
    client_socket.sendall(request.encode('utf-8'))

    # 接收服务器返回的提示
    response = client_socket.recv(4096).decode('utf-8')
    print("Server reply:", response)

    # 如果服务器提示需要输入数据，则按照要求输入数据
    if "Please enter data" in response:
        print("Please follow the prompts to enter data for 5 time steps, \n"
              "each time step containing 12 features \n"
              "(separated by commas, enter and wrap each time step)")
        lines = []
        for i in range(5):
            line = input("Please enter the data in line {}:".format(i + 1))
            lines.append(line)
        data_str = "\n".join(lines)
        client_socket.sendall(data_str.encode('utf-8'))

        # 接收服务器返回的预测结果
        final_response = client_socket.recv(4096).decode('utf-8')
        print("Server prediction result:", final_response)
    else:
        print("Server response:", response)

    client_socket.close()


if __name__ == "__main__":
    main()
