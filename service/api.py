from flask import Flask, request, render_template
import logging
import joblib
import numpy as np

app = Flask(__name__)

logging.basicConfig(
    filename='logs/log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Загрузка модели
model_path = './models/model.pkl'
with open(model_path, 'rb') as f:
    model = joblib.load(f)

def format_price(price):
    price = int(price)
    millions = price // 1_000_000
    thousands = (price % 1_000_000) // 1_000

    parts = []
    if millions > 0:
        parts.append(f"{millions} млн")
    if thousands > 0:
        parts.append(f"{thousands} тыс руб.")
    
    return ' '.join(parts) if parts else "менее тысячи рублей"

# Маппинг author_type из формы в числовой код для модели
AUTHOR_TYPE_MAP = {
    'агент по недвижимости': 3,
    'риэлтор': 4,
    'домовладелец': 0
}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            floor = int(request.form.get('floor', 0))
            floors_count = int(request.form.get('floors_count', 0))
            rooms_count = int(request.form.get('rooms_count', 0))
            total_meters = float(request.form.get('total_meters', 0))
            author_type_raw = request.form.get('author_type', '').lower()

            if floor <= 0 or floors_count <= 0 or total_meters <= 0:
                raise ValueError("Все числовые параметры должны быть положительными.")

            if author_type_raw not in AUTHOR_TYPE_MAP:
                raise ValueError("Недопустимое значение author_type")

            author_type_num = AUTHOR_TYPE_MAP[author_type_raw]

            features = np.array([[floor, floors_count, rooms_count, total_meters, author_type_num]])
            predicted_price = model.predict(features)[0]

            formatted_price = format_price(predicted_price)

            logging.info(f"Предсказание цены: floor={floor}, floors_count={floors_count}, roomss_count={rooms_count}, total_meters={total_meters}, author_type={author_type_raw} ({author_type_num}) → {predicted_price} руб. ({formatted_price})")

            return render_template('index.html', result=formatted_price,
                                   floor=floor, floors_count=floors_count, rooms_count=rooms_count,
                                   total_meters=total_meters, author_type=author_type_raw)

        except ValueError as e:
            logging.warning(f"Ошибка при вводе: {e}")
            return render_template('index.html', error=str(e))

        except Exception as e:
            logging.error(f"Ошибка при предсказании: {e}")
            return render_template('index.html', error="Внутренняя ошибка сервера.")

    return render_template('index.html')
    

if __name__ == '__main__':
    app.run(debug=True)
