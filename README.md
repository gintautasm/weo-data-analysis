### Sample data analysis

* root - analysis in `jupyter notebook`
  - mandatory cell
  - `File loading and imports`
  - other dependent cells are divided by major sections
* api - flask web api
* database not included it has to be downloaded manually
[`WEOOct2020all.xls`](https://www.imf.org/~/media/Files/Publications/WEO/WEO-Database/2020/02/WEOOct2020all.ashx)
[WEOO database](https://www.imf.org/en/Publications/WEO/weo-database/2020/October/download-entire-database)

### start project
```python
# create virtual env
cd api && python venv venv
# activate 
. api/venv/Scripts/activate
# install requirements
pip install -r requirements.txt
```

### run flask app
```
export FLASK_APP=api/app.py
flask run
```

### run flask tests
```
python -m pytest
```

### deactivating virtual environment
```python
deactivate
```