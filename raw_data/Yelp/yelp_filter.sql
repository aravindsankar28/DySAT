CREATE TABLE business_az as (select * from business where state='AZ');

CREATE TABLE category_az as (select id, business_id, category from category where business_id in (select id from business_az));

CREATE TABLE category_az_rest as (select * from category_az where category in ('Sandwiches', 'American (New)', 'Mexican', 'Italian', 'Chinese', 'Coffee & Tea', 'Ice Cream & Frozen Yogurt'));

# Filter so that only one category is avaiable.
CREATE TABLE az_bids as (select business_id, count(business_id) from category_az_rest group by business_id having count(business_id) = 1);

# az bids cats
CREATE TABLE az_bids_cats as (select az.business_id, category from az_bids as az, category_az_rest as c where az.business_id = c.business_id);

# az - reviews
CREATE TABLE az_reviews as (select r.user_id, r.business_id , r.date from review as r, az_bids_cats as f where r.business_id = f.business_id);

CREATE TABLE az_ub_filt as (select user_id, business_id, date from az_reviews where user_id in (select user_id from az_reviews group by user_id having count(user_id) >=10));

SET @directory = '/Users/aravind/Downloads/';

set @myvar = concat('SELECT * FROM az_ub_filt INTO OUTFILE ',"'",@directory, 'user_business.csv',"'");
PREPARE stmt1 FROM @myvar;
EXECUTE stmt1;
Deallocate prepare stmt1;

set @myvar = concat('SELECT * FROM az_bids_cats INTO OUTFILE ',"'",@directory, 'business_categories.csv',"'");

PREPARE stmt1 FROM @myvar;
EXECUTE stmt1;
Deallocate prepare stmt1;
