import cv2
from pyzbar.pyzbar import decode
import csv, os
from datetime import datetime
import winsound  # only works on Windows

cap = cv2.VideoCapture(0)  # try 0 if 1 doesn’t work

file_exists = os.path.isfile("Barcodes.csv")

with open("Barcodes.csv", mode="a", newline="") as file:
    writer = csv.writer(file)
    if not file_exists:
        writer.writerow(["Data", "Type", "Timestamp"])

    scanned = set()

    while True:
        success, frame = cap.read()
        if not success:
            break

        barcodes = decode(frame)

        for barcode in barcodes:
            x, y, w, h = barcode.rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            barcode_data = barcode.data.decode("utf-8")
            barcode_type = barcode.type
            text = f"{barcode_data} ({barcode_type})"
            cv2.putText(frame, text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if barcode_data not in scanned:
                scanned.add(barcode_data)
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                writer.writerow([barcode_data, barcode_type, now])
                print(f"[SAVED] {barcode_data} - {barcode_type}")
                winsound.Beep(1000, 150)  # beep on save

        cv2.imshow("Barcode Scanner", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
