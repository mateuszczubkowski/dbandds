import pandas as pd
import psycopg2
import psycopg2.extras
import numpy as np
import scipy.stats as sc
import datetime
from matplotlib import pyplot
import seaborn as sns
from sklearn.linear_model import LinearRegression


def create_tables():
    commands = (
        """
        CREATE TABLE employees (
            EmployeeId INTEGER PRIMARY KEY,
            FirstName VARCHAR(255) NOT NULL,
            MiddleInitial VARCHAR(1),
            LastName VARCHAR(255) NOT NULL
        )
        """,
        """
        CREATE TABLE customers (
            CustomerId INTEGER PRIMARY KEY,
            FirstName VARCHAR(255) NOT NULL,
            MiddleInitial VARCHAR(1),
            LastName VARCHAR(255) NOT NULL
        )
        """,
        """
        CREATE TABLE products (
            ProductId INTEGER PRIMARY KEY,
            Name VARCHAR(255) NOT NULL,
            Price NUMERIC NOT NULL
        )
        """,
        """
        CREATE TABLE sales (
                SalesId INTEGER PRIMARY KEY,
                SalesPersonId INTEGER REFERENCES employees (EmployeeId),
                CustomerId INTEGER REFERENCES customers (CustomerId),
                ProductId INTEGER REFERENCES products (ProductId),
                Quantity INTEGER NOT NULL,
                SalesDate DATE NOT NULL
        )
        """)

    try:
        conn = psycopg2.connect(
            host="localhost",
            database="dbandds",
            user="postgres",
            password="postgres")
        cur = conn.cursor()

        for command in commands:
            cur.execute(command)

        cur.close()
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()


def create_stat_table():
    command = """
        CREATE TABLE stats_table (
                StatsName VARCHAR(30) NOT NULL,
                StatsDate DATE NOT NULL,
                Mean NUMERIC NOT NULL,
                Median NUMERIC NOT NULL,
                Min NUMERIC NOT NULL,
                Max NUMERIC NOT NULL,
                Std NUMERIC NOT NULL,
                Variance NUMERIC NOT NULL,
                Skewness NUMERIC NOT NULL,
                Kurtosis NUMERIC NOT NULL
        )
        """
    try:
        conn = psycopg2.connect(
            host="localhost",
            database="dbandds",
            user="postgres",
            password="postgres")
        cur = conn.cursor()

        cur.execute(command)

        cur.close()
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()


# Dane z CSV do bazy
def load_data_to_database():
    conn = psycopg2.connect(
                host="localhost",
                database="dbandds",
                user="postgres",
                password="postgres")
    # Employees
    cur = conn.cursor()
    with open('Employees.csv', 'r') as f:
        cur.copy_from(f, 'employees', sep=',')
    # Products
    cur = conn.cursor()
    with open('Products.csv', 'r') as f:
        cur.copy_from(f, 'products', sep=',')
    # Customers
    cur = conn.cursor()
    with open('Customers.csv', 'r') as f:
        cur.copy_from(f, 'customers', sep=',')
    # Sales
    cur = conn.cursor()
    with open('Sales.csv', 'r') as f:
        cur.copy_from(f, 'sales', sep=',')

    conn.commit()


# Dane z bazy do pandas dataframe
def load_data_from_database():
    conn = psycopg2.connect(
        host="localhost",
        database="dbandds",
        user="postgres",
        password="postgres")

    employees = pd.read_sql("select * from \"employees\"", conn)
    products = pd.read_sql("select * from \"products\"", conn)
    customers = pd.read_sql("select * from \"customers\"", conn)
    sales = pd.read_sql("select * from \"sales\"", conn)

    conn.close()
    return employees, products, customers, sales


def get_stats(column):
    val_max = column.max(axis=0)
    val_min = column.min(axis=0)
    val_mean = column.mean(axis=0)
    val_med = column.median(axis=0)
    val_stdev = column.std(axis=0)
    val_var = column.var(axis=0)
    val_kurt = column.kurtosis(axis=0)
    val_skew = column.skew(axis=0)
    return val_max, val_min, val_mean, val_med, val_stdev, val_var, val_kurt, val_skew


def insert_into_database(data):
    try:
        conn = psycopg2.connect(
            host="localhost",
            database="dbandds",
            user="postgres",
            password="postgres")
        cur = conn.cursor()

        cur = conn.cursor()
        insert_query = 'insert into stats_table values %s'
        psycopg2.extras.execute_values(cur, insert_query, data, template=None)

        cur.close()
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()


# create_tables()
# load_data_to_database()
employees, products, customers, sales = load_data_from_database()
print(employees.head())
print(products.head())
print(customers.head())
print(sales.head())

# Statystyki podstawowe
# Dodajemy do Sales kwotę sprzedaży
sales = pd.merge(sales, products[["productid", "price"]], how='left', left_on='productid', right_on='productid')
sales["gains"] = sales["quantity"] * sales["price"]
sales = sales.drop(columns=["price"])

gains_max, gains_min, gains_mean, gains_med, gains_stdev, gains_var, gains_kurt, gains_skew = get_stats(sales["gains"])
print("Sales gains stats:")
print("Max: ", gains_max)
print("Min: ", gains_min)
print("Mean: ", gains_mean)
print("Median: ", gains_med)
print("Standard deviation: ", gains_stdev)
print("Variance: ", gains_var)
print("Kurtosis: ", gains_kurt)
print("Skewness: ", gains_skew)

# Dodajemy do Employees liczbę sprzedaży
employees["salesnumber"] = employees["employeeid"].map(sales["salespersonid"].value_counts())
empl_sales_max, empl_sales_min, empl_sales_mean, empl_sales_med, empl_sales_stdev, empl_sales_var, empl_sales_kurt, empl_sales_skew = get_stats(employees["salesnumber"] )
print("Sales per employee stats:")
print("Max: ", empl_sales_max)
print("Min: ", empl_sales_min)
print("Mean: ", empl_sales_mean)
print("Median: ", empl_sales_med)
print("Standard deviation: ", empl_sales_stdev)
print("Variance: ", empl_sales_var)
print("Kurtosis: ", empl_sales_kurt)
print("Skewness: ", empl_sales_skew)

# Dodajemy do customers wydaną kwotę
customers_temp = sales.groupby(["customerid"]).sum().reset_index()
customers = pd.merge(customers, customers_temp[["customerid", "gains"]], how='outer', left_on='customerid', right_on='customerid')
customers["amountspent"] = customers["gains"]
customers = customers.drop(columns=["gains"])
customers["amountspent"] = customers["amountspent"].fillna(0)

c_spent_max, c_spent_min, c_spent_mean, c_spent_med, c_spent_stdev, c_spent_var, c_spent_kurt, c_spent_skew = get_stats(customers["amountspent"])
print("Buys per customer stats:")
print("Max: ", c_spent_max)
print("Min: ", c_spent_min)
print("Mean: ", c_spent_mean)
print("Median: ", c_spent_med)
print("Standard deviation: ", c_spent_stdev)
print("Variance: ", c_spent_var)
print("Kurtosis: ", c_spent_kurt)
print("Skewness: ", c_spent_skew)

# Dane statystyczne sprzedaży podzielone na miesiące, następnie zapiszemy je do bazy danych
# create_stat_table()
sales["salesdate"] = pd.to_datetime(sales["salesdate"])
firstDate = sales["salesdate"].min()
lastDate = sales["salesdate"].max()
values_list = []
for y in range(firstDate.year, lastDate.year + 1, 1):
    for m in range(1, 13, 1):
        if not(y == lastDate.year and m <= lastDate.month):
            mask = (sales['salesdate'].dt.year == y) & (sales['salesdate'].dt.month == m)
            date = datetime.datetime(y, m, 1)
            v_max, v_min, v_mean, v_med, v_stdev, v_var, v_kurt, v_skew = get_stats(sales.loc[mask]["gains"])
            values_list.append(("sales-gains", date, v_mean, v_med, v_min, v_max, v_stdev, v_var, v_skew, v_kurt))

#insert_into_database(values_list)

# Graficzne przedstawienie normalności rozkładu dla wysokości zysków ze sprzedaży
pyplot.hist(sales["gains"])
pyplot.show()

# Badanie normalności rozkładu

# Shapiro-Wilk
stat, p = sc.shapiro(sales["gains"])
print("Stat= ", stat, " p= ", p)
alpha = 0.05
if p > alpha:
    print("Rozkład normalny")
else:
    print("Rozkład różni się od normalanego")

# duża liczba rekordów i możliwe że dlatego p wyszło 0, na dole dla losowych 5000 próbek
stat, p = sc.shapiro(sales["gains"].sample(5000))
print("Stat= ", stat, " p= ", p)
alpha = 0.05
if p > alpha:
    print("Rozkład normalny")
else:
    print("Rozkład różni się od normalanego")

# D'Agostino
stat, p = sc.normaltest(sales["gains"])
print("Stat= ", stat, " p= ", p)
alpha = 0.05
if p > alpha:
    print("Rozkład normalny")
else:
    print("Rozkład różni się od normalanego")

# testy Shapiro-Wilk'a i D'Agostino dla innej zmiennej, ilość dokonanych sprzedazy
stat, p = sc.shapiro(employees["salesnumber"])
print("Stat= ", stat, " p= ", p)
alpha = 0.05
if p > alpha:
    print("Rozkład normalny")
else:
    print("Rozkład różni się od normalanego")

# D'Agostino
stat, p = sc.normaltest(employees["salesnumber"])
print("Stat= ", stat, " p= ", p)
alpha = 0.05
if p > alpha:
    print("Rozkład normalny")
else:
    print("Rozkład różni się od normalanego")


# testy równości wariancji, podzelimy sprzedaze na dwie grupy wedlug lat - 2017 i 2018 vs 2019 i 2020
sales17_18 = sales[lambda x: x["salesdate"].dt.year < 2019]
sales19_20 = sales[lambda x: x["salesdate"].dt.year >= 2019]
test_var = sc.levene(sales17_18["gains"], sales19_20["gains"])
print('Statystyka T i prawdopodobienstwo dla wariancji: ', test_var)

# test równości średnich dla powyższych grup
test_mean = sc.ttest_ind(sales17_18["gains"], sales19_20["gains"])
print('Statystyka T i prawdopodobienstwo dla średnich: ', test_mean)

# dodamy 2 kolumny do tabeli pracownicy, zarobki oraz staz pracy, aby zbadac wplyw zmiennych na inne
# niestety będą to losowe liczby, więc prawdopodobnie nie będą one miały na siebie wpływu
employees["seniority"] = np.random.randint(2, 8, employees.shape[0])
salaries = np.arange(2800, 4100, 100).tolist()
employees["salary"] = np.random.choice(salaries, employees.shape[0])

corr_df = employees[["seniority", "salary", "salesnumber"]].corr()
sns.heatmap(corr_df, xticklabels=corr_df.columns.values, yticklabels=corr_df.columns.values, annot=True)
heat_map = pyplot.gcf()
heat_map.set_size_inches(10, 5)
pyplot.xticks(fontsize=10)
pyplot.yticks(fontsize=10)
pyplot.show()

# regresja liniowa
# czy staz pracy ma wplyw na zarobki?
x = employees.seniority.to_numpy().reshape((-1,1))
y = employees.salary.to_numpy()

model = LinearRegression()
model.fit(x, y)

print('Wartość wyrazu wolnego = ', model.intercept_)
print('Wartość współczynnika przy stażu pracy = ', model.coef_)
print('Współczynnik determinacji R^2 = ', model.score(x,y))
print('Wartości przewidywane dla zarobków: ', model.predict(x))

# czy liczba sprzedazy ma wplyw na zarobki?
employees.fillna(method='ffill', inplace=True)
x = employees.salesnumber.to_numpy().reshape((-1,1))
y = employees.salary.to_numpy()
model = LinearRegression()
model.fit(x, y)

print('Wartość wyrazu wolnego = ', model.intercept_)
print('Wartość współczynnika przy liczbie sprzedazy = ', model.coef_)
print('Współczynnik determinacji R^2 = ', model.score(x,y))
print('Wartości przewidywane dla zarobków: ', model.predict(x))

# czy obie te zmienne mają wpływ na zarobki?
pairs = []

for i in range(len(employees.seniority)):
    pair = []
    pair.append(employees.seniority[i])
    pair.append(employees.salesnumber[i])
    pairs.append(pair)

x = np.array(pairs)
y = employees.salary.to_numpy()

model = LinearRegression()
model.fit(x, y)

print('Wartość wyrazu wolnego = ', model.intercept_)
print('Wartość współczynnika przy x = ', model.coef_)
print('Współczynnik determinacji R^2 = ', model.score(x,y))


def add_employee(firstname, middleinitial, lastname):
    conn = None
    try:
        conn = psycopg2.connect(
            host="localhost",
            database="dbandds",
            user="postgres",
            password="postgres")
        cur = conn.cursor()

        cur.execute('CALL add_employee(%s,%s,%s)', (firstname, middleinitial, lastname))

        conn.commit()

        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()


def add_customer(firstname, middleinitial, lastname):
    conn = None
    try:
        conn = psycopg2.connect(
            host="localhost",
            database="dbandds",
            user="postgres",
            password="postgres")
        cur = conn.cursor()

        cur.execute('CALL add_customer(%s,%s,%s)', (firstname, middleinitial, lastname))

        conn.commit()

        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()


def add_sale(salespersonid, customerid, productid, quantity, salesdate):
    conn = None
    try:
        conn = psycopg2.connect(
            host="localhost",
            database="dbandds",
            user="postgres",
            password="postgres")
        cur = conn.cursor()

        cur.execute('CALL add_sale(%s,%s,%s,%s,%s)', (salespersonid, customerid, productid, quantity, salesdate))

        conn.commit()

        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()


def add_product(name, price):
    conn = None
    try:
        conn = psycopg2.connect(
            host="localhost",
            database="dbandds",
            user="postgres",
            password="postgres")
        cur = conn.cursor()

        cur.execute('CALL add_product(%s,%s)', (name, price))

        conn.commit()

        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()


def edit_sale(id, salespersonid, customerid, productid, quantity, salesdate):
    conn = None
    try:
        conn = psycopg2.connect(
            host="localhost",
            database="dbandds",
            user="postgres",
            password="postgres")
        cur = conn.cursor()

        cur.execute('CALL edit_sale(%s,%s,%s,%s,%s,%s)', (id, salespersonid, customerid, productid, quantity, salesdate))

        conn.commit()

        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()


def edit_product(id, name, price):
    conn = None
    try:
        conn = psycopg2.connect(
            host="localhost",
            database="dbandds",
            user="postgres",
            password="postgres")
        cur = conn.cursor()

        cur.execute('CALL edit_product(%s,%s,%s)', (id, name, price))

        conn.commit()

        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()


def edit_employee(id, firstname, middleinitial, lastname):
    conn = None
    try:
        conn = psycopg2.connect(
            host="localhost",
            database="dbandds",
            user="postgres",
            password="postgres")
        cur = conn.cursor()

        cur.execute('CALL edit_employee(%s,%s,%s,%s)', (id, firstname, middleinitial, lastname))

        conn.commit()

        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()


def edit_customer(id, firstname, middleinitial, lastname):
    conn = None
    try:
        conn = psycopg2.connect(
            host="localhost",
            database="dbandds",
            user="postgres",
            password="postgres")
        cur = conn.cursor()

        cur.execute('CALL edit_customer(%s,%s,%s,%s)', (id, firstname, middleinitial, lastname))

        conn.commit()

        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()


def delete_sale(id):
    conn = None
    try:
        conn = psycopg2.connect(
            host="localhost",
            database="dbandds",
            user="postgres",
            password="postgres")
        cur = conn.cursor()

        cur.execute('CALL delete_sale(%s)', (id))

        conn.commit()

        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()


def delete_product(id):
    conn = None
    try:
        conn = psycopg2.connect(
            host="localhost",
            database="dbandds",
            user="postgres",
            password="postgres")
        cur = conn.cursor()

        cur.execute('CALL delete_product(%s)', (id))

        conn.commit()

        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()


def delete_employee(id):
    conn = None
    try:
        conn = psycopg2.connect(
            host="localhost",
            database="dbandds",
            user="postgres",
            password="postgres")
        cur = conn.cursor()

        cur.execute('CALL delete_employee(%s)', (id))

        conn.commit()

        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()


def delete_customer(id):
    conn = None
    try:
        conn = psycopg2.connect(
            host="localhost",
            database="dbandds",
            user="postgres",
            password="postgres")
        cur = conn.cursor()

        cur.execute('CALL delete_customer(%s)', (id))

        conn.commit()

        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()


# add_product('test', 9)