import pandas as pd
import joblib  # ใช้สำหรับโหลดโมเดลที่บันทึกไว้

# 1. โหลดโมเดลและ TF-IDF Vectorizer
model = joblib.load('model.pkl')  # โหลดโมเดลที่บันทึกไว้
vectorizer = joblib.load('vectorizer.pkl')  # โหลด TF-IDF Vectorizer ที่ใช้ตอนเทรน

# 2. โหลดข้อมูล test.csv
test_data = pd.read_csv('test.csv')  # สมมติว่ามีคอลัมน์ชื่อ 'text' ที่เป็นข้อความข่าว
test_data = test_data.dropna()  # ลบข้อมูลที่เป็นค่าว่าง

# 3. ตรวจสอบว่ามีคอลัมน์ 'id' และ 'text' หรือไม่
if 'id' not in test_data.columns:
    raise ValueError("Error: คอลัมน์ 'id' ไม่พบในไฟล์ test.csv")
if 'text' not in test_data.columns:
    raise ValueError("Error: คอลัมน์ 'text' ไม่พบในไฟล์ test.csv")

# 4. ลบแถวที่คอลัมน์ 'id' ไม่ใช่ตัวเลข
test_data['id'] = pd.to_numeric(test_data['id'], errors='coerce')  # แปลง 'id' เป็นตัวเลข ถ้าไม่ได้จะเป็น NaN
test_data = test_data.dropna(subset=['id'])  # ลบแถวที่ 'id' เป็น NaN

# 5. แปลงข้อความใน test.csv ด้วย TF-IDF Vectorizer
X_test = vectorizer.transform(test_data['text'])  # แปลงข้อความเป็นเวกเตอร์

# 6. ใช้โมเดลที่โหลดมาทำนายผล
predictions = model.predict(X_test)  # ทำนายผล (0 = Real News, 1 = Fake News)

# 7. เพิ่มผลลัพธ์ลงใน DataFrame
test_data['prediction'] = predictions  # เพิ่มคอลัมน์ 'prediction' ลงในไฟล์ test.csv

# 8. บันทึกผลลัพธ์กลับไปที่ไฟล์ CSV
test_data.to_csv('predictions.csv', index=False)  # บันทึกเป็นไฟล์ predictions.csv
print("Results saved to predictions.csv!")