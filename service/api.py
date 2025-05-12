from flask import Flask, request, render_template
import logging

app = Flask(__name__)

# Настройка логгера
logging.basicConfig(
    filename='logs/log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def format_price(price):
    millions = int(price) // 1_000_000
    thousands = (int(price) % 1_000_000) // 1_000

    parts = []
    if millions > 0:
        parts.append(f"{millions} млн")
    if thousands > 0:
        parts.append(f"{thousands} тыс руб.")
    
    return ' '.join(parts) if parts else "менее тысячи рублей"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            sqm = float(request.form.get('square_meters', 0))
            if sqm <= 0:
                raise ValueError("Площадь должна быть положительной.")

            price = sqm * 300_000
            formatted_price = format_price(price)
            logging.info(f"Расчет стоимости: {sqm} кв.м → {price} руб. ({formatted_price})")
            return render_template('index.html', result=formatted_price, sqm=sqm)

        except ValueError as e:
            logging.warning(f"Ошибка при вводе: {e}")
            return render_template('index.html', error=str(e))

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
