from flask import Flask, render_template, request, redirect, flash
import cv2
import numpy as np
from ultralytics import YOLO
import base64
from werkzeug.utils import secure_filename
import os
from flask import send_from_directory

app = Flask(__name__)
app.secret_key = 'my_secret_key123'

# Tải mô hình YOLO đã được huấn luyện trước
model = YOLO("best.pt")

# Đường dẫn tới thư mục tạm cho tệp tin tải lên
app.config['UPLOAD_FOLDER'] = 'uploads'
# Định rõ phương thức lưu trữ tệp tin
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  


@app.route('/result/<filename>')
def serve_result_image(filename):
    return send_from_directory('static', filename)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Kiểm tra xem có tệp tin được tải lên không
        if 'image' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['image']

        # Kiểm tra xem tệp tin có tên không
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        # Xử lý tệp tin tải lên
        if file:
            # Lưu tệp tin vào thư mục tạm thời
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            # Thực hiện đếm đối tượng trên ảnh
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            object_count = perform_object_count(image_path)

            # Truyền giá trị của object_count vào index.html
            result_image_path = f'result/{filename}'
            return render_template('index.html', result_image_path=result_image_path, object_count=object_count)
            # return render_template('index.html', object_count=object_count)
    return render_template('index.html')


def perform_object_count(image_path):
    # Đọc ảnh từ đường dẫn
    image = cv2.imread(image_path)

    # Thay đổi kích thước ảnh để vừa với 640x640 và vẫn giữ tỷ lệ khung hình
    height, width, _ = image.shape
    max_size = max(height, width)
    scale = 640 / max_size
    resized_image = cv2.resize(image, None, fx=scale, fy=scale)

    # Thực hiện phát hiện đối tượng
    predictions = model.predict(resized_image)

    # Khởi tạo đếm đối tượng
    object_count = 0

    # Vẽ khung giới hạn trên ảnh đã thay đổi kích thước và hiển thị nhãn
    for prediction in predictions:
        for box in prediction.boxes:
            x_min, y_min, x_max, y_max = box.xyxy[0]  # Trích xuất tọa độ khung giới hạn
            conf = box.conf[0]  # Trích xuất độ tin cậy

            # Chuyển đổi tọa độ khung giới hạn thành số nguyên
            x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

            # Nếu độ tin cậy lớn hơn ngưỡng
            if conf > 0.8:
                cv2.rectangle(resized_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                labelcount = f"{object_count + 1}"
                text_size, _ = cv2.getTextSize(labelcount, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

                text_x = int((x_min + x_max) / 2 - text_size[0] / 2)
                text_y = int((y_min + y_max) / 2 + text_size[1] / 2)
                cv2.putText(resized_image, labelcount, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            object_count += 1

    # Hiển thị số lượng đối tượng
    # count_label = f"CountingSteel: {object_count}"
    # cv2.putText(resized_image, count_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Lưu ảnh đã thay đổi kích thước vào thư mục static
    result_image_path = os.path.join('static', 'resized_image.png')
    cv2.imwrite(result_image_path, resized_image)

    return object_count


if __name__ == '__main__':
    app.run(debug=True)
