import cv2
import numpy as np
import base64

def preprocess_image(image_data):
    # Decode base64 image
    image_data = image_data.split(",")[1]
    img_bytes = base64.b64decode(image_data)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
    
    # 1. Check if background is light and invert if needed to make it a white digit on a black background
    border_pixels = np.concatenate([img[0, :], img[-1, :], img[:, 0], img[:, -1]])
    if np.mean(border_pixels) > 127:
        img = cv2.bitwise_not(img)
        
    # 2. Crop the image to the bounding box of the digit
    # Find all non-zero pixels (foreground is white, i.e. > 0)
    pts = np.argwhere(img > 10)
    if len(pts) > 0:
        y1, x1 = pts.min(axis=0)
        y2, x2 = pts.max(axis=0)
        
        # Add a small padding to the crop
        h_crop, w_crop = y2 - y1 + 1, x2 - x1 + 1
        pad = max(4, int(0.08 * max(h_crop, w_crop)))
        y1 = max(0, y1 - pad)
        y2 = min(img.shape[0] - 1, y2 + pad)
        x1 = max(0, x1 - pad)
        x2 = min(img.shape[1] - 1, x2 + pad)
        
        cropped = img[y1:y2+1, x1:x2+1]
        
        # 3. Resize cropped image to fit in 20x20 box while preserving aspect ratio
        h, w = cropped.shape
        if h > w:
            new_h = 20
            new_w = int(round(w * (20.0 / h)))
            new_w = max(1, new_w)
        else:
            new_w = 20
            new_h = int(round(h * (20.0 / w)))
            new_h = max(1, new_h)
            
        resized_digit = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 4. Center the resized digit inside a blank 28x28 background using Center of Mass (Moments)
        padded_digit = np.zeros((28, 28), dtype=np.uint8)
        y_start = (28 - new_h) // 2
        x_start = (28 - new_w) // 2
        padded_digit[y_start:y_start+new_h, x_start:x_start+new_w] = resized_digit
        
        # Calculate center of mass of the padded digit
        M = cv2.moments(padded_digit)
        if M["m00"] > 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            
            # Translate to shift center of mass (cx, cy) to target center (14, 14)
            dx = int(round(14.0 - cx))
            dy = int(round(14.0 - cy))
            
            M_trans = np.float32([[1, 0, dx], [0, 1, dy]])
            img = cv2.warpAffine(padded_digit, M_trans, (28, 28), flags=cv2.INTER_AREA)
        else:
            img = padded_digit
    else:
        # Fallback to standard resize if canvas is blank
        img = cv2.resize(img, (28, 28))

    img = img / 255.0
    img = img.reshape(1, 28, 28, 1)
    return img
