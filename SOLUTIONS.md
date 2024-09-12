He creado dos ficheros, que corresponden a las funcionalidades dadas por los dos notebooks en lab\analysis. Estos dos ficheros con formato .py estructuran todas las funciones de tratamiento y procesamiento de datos, de ejecución, entrenamiento y validación del clasificador Random Forest y de visualización de gráficas.

Por consiguiente, he creado la API (app.py) para definir la categoría de precio a un listado de valores dados a las características. Esta API opera en localhost y en el puerto 8000 y he utilizado las librerías uvicorn y FastAPI. La inicializo con el comando 

>> uvicorn app:app --host 0.0.0.0 --port 8000 --reload

También, dejo también la opción de inicializarla desde el código.

El resultado de ejecutar el comando de POST:

>> Invoke-RestMethod -Uri 'http://127.0.0.1:8000/predict/' `
>>   -Method 'Post' `
>>   -Headers @{ "Content-Type" = "application/json" } `
>>   -Body '{
>>     "id": 1001,
>>     "accommodates": 4,
>>     "room_type": "Entire home/apt",
>>     "beds": 2,
>>     "bedrooms": 1,
>>     "bathrooms": 2,
>>     "neighbourhood": "Brooklyn",
>>     "tv": 1,
>>     "elevator": 1,
>>     "internet": 0,
>>     "latitude": 40.71383,
>>     "longitude": -73.9658
>> }'

Proporciona la siguiente respuesta:   

>> "id price_category 1001 High"

Por lo tanto, el modelo está dando los resultados esperados.


Por último, he dockerizado la api y he utilizado los siguientes comandos:

>> docker build -t api .
>> 
>> docker run -p 8000:8000 api

Se comprueba que está activo el contenedor:

>> CONTAINER ID   IMAGE     COMMAND                  CREATED         STATUS              PORTS                         
>> 56b6359fe9c2   api       "uvicorn app:app --h…"   2 minutes ago   Up About a minute   0.0.0.0:8000->8000/tcp   
