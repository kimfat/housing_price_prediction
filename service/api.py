from flask import Flask, request, render_template
import logging

app = Flask(__name__)

# Настройка логгера
logging.basicConfig(
    filename='logs/log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            sqm = float(request.form.get('square_meters', 0))
            if sqm <= 0:
                raise ValueError("Площадь должна быть положительной.")

            price = int(sqm * 300000)
            logging.info(f"Расчет стоимости: {sqm} кв.м → {price} руб.")
            return render_template('index.html', result=price, sqm=sqm)

        except ValueError as e:
            logging.warning(f"Ошибка при вводе: {e}")
            return render_template('index.html', error=str(e))

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)