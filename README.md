# ML_HW1
Репозиторий представляет собой выполненное домашнее задание по курсу ML от HSE
В данной работе было сделано следующее:
1. Проведён предварительный анализ датафрейма
2. Датафрейм отредачен и приведён в нормальный вид, без NaN и повторяющихся строк
3. Визуализированы некоторые зависимости датафрейма
4. Обучены несколько моделей линейной регрессии, L1, R2, вычислены функционалы ошибок для них. Как оказалось, стандартизация фич даёт очень большой буст верных предиктов. Также, буст верных предиктов даёт подбор оптимальных параметров различных моделей с помощью метода перебора по сетке (GridSearchCV).
5. Были обновлены старые фичи (корень из year)и были добавлены дополнительные фичи, включая те, которые были добыты с помощью парсинга Wikipedia, также проведено сравнение двух моделей линейной регрессии (вторая с новыми фичами), в результате вторая оказалась чуть лучше
6. Реализован веб-сервис, позволяющий получить предсказание модели по входным данным (json файл с параметрами машины). Сервис деплоился на render.com
   Ниже прикреплены скриншоты успешной работы сервиса FastAPI. Тесты проводились с помощью Postman
   На этом скриншоте изображен метод predict_item, получающий на вход json машины и возвращающий предсказание цены для неё
   ![image](https://github.com/MrseFuntik/ML_HW1/assets/136927535/707eea2f-8454-47e0-a171-7ea04e1a22b5)

   На этом скриншоте изображен метод predict_items, получающий на вход список с json-строками машин и возвращающий обновленный json-список, с дополнительным столбцом, содержащим предсказания для каждой из машин
   ![image](https://github.com/MrseFuntik/ML_HW1/assets/136927535/ac6225de-3bae-40cf-91cf-474e22632ea7)

Ссылка на поднятый веб-сервис: https://ml-hw1-fastapi-12tr.onrender.com
Если будете тестировать запросы FastAPI, нужно будет подождать пару минут, пока сервис рестартнется
