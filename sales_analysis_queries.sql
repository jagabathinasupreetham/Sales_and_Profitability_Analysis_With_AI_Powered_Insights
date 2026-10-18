-- Sales & Profitability Analytics
CREATE DATABASE IF NOT EXISTS sales_analytics;
USE sales_analytics;

--CREATING TABLE
CREATE TABLE IF NOT EXISTS cleaned_sales_data (
    row_id INT PRIMARY KEY,
    order_id VARCHAR(30),
    order_date DATE,
    ship_date DATE,
    ship_mode VARCHAR(30),
    customer_id VARCHAR(30),
    customer_name VARCHAR(100),
    segment VARCHAR(30),
    city VARCHAR(100),
    state VARCHAR(100),
    country VARCHAR(100),
    market VARCHAR(30),
    region VARCHAR(50),
    product_id VARCHAR(40),
    category VARCHAR(50),
    `sub-category` VARCHAR(50),
    product_name VARCHAR(255),
    sales DECIMAL(12, 5),
    quantity INT,
    discount DECIMAL(6, 3),
    profit DECIMAL(12, 5),
    shipping_cost DECIMAL(12, 3),
    order_priority VARCHAR(20)
);

--IMPORTING THE CSV
LOAD DATA LOCAL INFILE '/Users/supreethamjagabathina/GitHub/Sales & Profitability Analytics with AI-Powered Insights/cleaned_sales_data.csv'
REPLACE INTO TABLE cleaned_sales_data
CHARACTER SET utf8mb4
FIELDS TERMINATED BY ','
OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(
    row_id,
    order_id,
    order_date,
    ship_date,
    ship_mode,
    customer_id,
    customer_name,
    segment,
    city,
    state,
    country,
    market,
    region,
    product_id,
    category,
    `sub-category`,
    product_name,
    sales,
    quantity,
    discount,
    profit,
    shipping_cost,
    order_priority
);


-- Quick import check
SELECT COUNT(*) AS total_rows_in_table
FROM cleaned_sales_data;

-- BUSINESS ANALYSIS QUERIES
-- 1. Overall sales performance: totals, distinct orders, and average sales per order.
SELECT
    ROUND(SUM(sales), 2) AS total_sales,
    ROUND(SUM(profit), 2) AS total_profit,
    COUNT(DISTINCT order_id) AS total_orders,
    ROUND(SUM(sales) / NULLIF(COUNT(DISTINCT order_id), 0), 2) AS average_sales_per_order
FROM cleaned_sales_data;

-- 2. Sales and profit by product category.
SELECT
    category,
    ROUND(SUM(sales), 2) AS total_sales,
    ROUND(SUM(profit), 2) AS total_profit
FROM cleaned_sales_data
GROUP BY category
ORDER BY total_sales DESC;

-- 3. Sales and profit by region.
SELECT
    region,
    ROUND(SUM(sales), 2) AS total_sales,
    ROUND(SUM(profit), 2) AS total_profit
FROM cleaned_sales_data
GROUP BY region
ORDER BY total_sales DESC;

-- 4. Monthly sales and profit, ordered chronologically.
SELECT
    DATE_FORMAT(order_date, '%Y-%m') AS order_month,
    ROUND(SUM(sales), 2) AS total_sales,
    ROUND(SUM(profit), 2) AS total_profit
FROM cleaned_sales_data
GROUP BY DATE_FORMAT(order_date, '%Y-%m')
ORDER BY order_month;

-- 5. Top 10 products by sales.
SELECT
    product_name,
    ROUND(SUM(sales), 2) AS total_sales,
    ROUND(SUM(profit), 2) AS total_profit
FROM cleaned_sales_data
GROUP BY product_name
ORDER BY total_sales DESC
LIMIT 10;

-- 6. Products with negative total profit.
SELECT
    product_name,
    ROUND(SUM(sales), 2) AS total_sales,
    ROUND(SUM(profit), 2) AS total_profit
FROM cleaned_sales_data
GROUP BY product_name
HAVING SUM(profit) < 0
ORDER BY total_profit ASC;

-- 7. Sales and profit by discount value.
SELECT
    discount,
    ROUND(SUM(sales), 2) AS total_sales,
    ROUND(SUM(profit), 2) AS total_profit,
    COUNT(*) AS row_count
FROM cleaned_sales_data
GROUP BY discount
ORDER BY discount;

-- 8. Sales and profit by customer segment.
SELECT
    segment,
    ROUND(SUM(sales), 2) AS total_sales,
    ROUND(SUM(profit), 2) AS total_profit,
    COUNT(DISTINCT customer_id) AS distinct_customers
FROM cleaned_sales_data
GROUP BY segment
ORDER BY total_sales DESC;

-- 9. Orders, sales, profit, and shipping cost by shipping mode.
SELECT
    ship_mode,
    COUNT(DISTINCT order_id) AS total_orders,
    ROUND(SUM(sales), 2) AS total_sales,
    ROUND(SUM(profit), 2) AS total_profit,
    ROUND(SUM(shipping_cost), 2) AS total_shipping_cost
FROM cleaned_sales_data
GROUP BY ship_mode
ORDER BY total_sales DESC;

-- 10. Sales and profit by country.
SELECT
    country,
    ROUND(SUM(sales), 2) AS total_sales,
    ROUND(SUM(profit), 2) AS total_profit
FROM cleaned_sales_data
GROUP BY country
ORDER BY total_sales DESC;

-- 11. Sales and profit by product subcategory.
SELECT
    `sub-category`,
    ROUND(SUM(sales), 2) AS total_sales,
    ROUND(SUM(profit), 2) AS total_profit
FROM cleaned_sales_data
GROUP BY `sub-category`
ORDER BY total_profit DESC;
