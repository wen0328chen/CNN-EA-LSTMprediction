import socket
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# 加载模型和 scaler
model = load_model('./model/CNN+EA-LSTM_best_model_5.h5')
scaler = joblib.load('scaler/scaler.pkl')
scaler_y = joblib.load('scaler/scaler_y.pkl')


def standardize_data(data):
    """
    对 data 中每个时间步的前 8 个特征进行标准化
    data: shape (5, 12)
    """
    data_copy = data.copy()
    # 假设 scaler 是按列处理，data_copy[:, :8] 的 shape 为 (5, 8)
    data_copy[:, :8] = scaler.transform(data_copy[:, :8])
    return data_copy


def handle_client(conn):
    # 接收 client 的初始请求信息
    req = conn.recv(1024).decode('utf-8')
    print("收到 client 请求：", req)

    # 简单判断是否包含“预测”
    if "predict" in req:
        prompt = ("Okay, please enter the 12 feature data of the current and previous 4 hours (a total of 5 time steps) in sequence,\n"
                  'pollution,humidity, temperature, pressure, wind direction(NE: 0, NW: 1, SE: 2 ,SW: 3), \n'
                  'wind speed, snow, rain, year, day, hour, month\n'
                  "Separate each feature with a comma, and separate the features of each time step with a carriage return. For example:\n"
                  "t1_1,t1_2,...,t1_12\n"
                  "t2_1,t2_2,...,t2_12\n"
                  "...\n"
                  "Please enter data:")
        conn.sendall(prompt.encode('utf-8'))

        # 接收数据（这里简单处理，假设所有数据一次性发送过来）
        data_str = conn.recv(4096).decode('utf-8')
        try:
            # 将数据按行分割，每行对应一个时间步
            lines = data_str.strip().splitlines()
            if len(lines) != 5:
                conn.sendall("Data format error: Must contain data for 5 time steps.".encode('utf-8'))
                return
            data_list = []
            for line in lines:
                # 每个时间步有12个特征
                features = [float(x) for x in line.split(',')]
                if len(features) != 12:
                    conn.sendall("Data format error: Each time step must have 12 features.".encode('utf-8'))
                    return
                data_list.append(features)
            data_array = np.array(data_list)  # shape (5, 12)
        except Exception as e:
            conn.sendall(("Data parsing failed:" + str(e)).encode('utf-8'))
            return

        # 对每个时间步的前8个特征进行标准化
        data_array = standardize_data(data_array)

        # 构造模型输入 (batch_size=1, time_steps=5, features=12)
        model_input = np.expand_dims(data_array, axis=0)

        # 模型预测，输出下一个时间步的结果
        pred = model.predict(model_input)
        #去标准化
        result_array = np.array(pred).reshape(1, -1)
        original_result = scaler_y.inverse_transform(result_array)
        # 假设模型输出形状为 (1, 1)，取第一个元素
        result = original_result[0, 0]
        reply = "The PM2.5 content in the air after one hour is:{:.2f}".format(result)
        conn.sendall(reply.encode('utf-8'))
    else:
        conn.sendall("Unrecognized request.".encode('utf-8'))


def main():
    host = ''
    port = 12345
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(5)
    print("Server startup, listening {}:{}".format(host, port))

    while True:
        conn, addr = server_socket.accept()
        print("Connection from:", addr)
        handle_client(conn)
        conn.close()


if __name__ == "__main__":
    main()
