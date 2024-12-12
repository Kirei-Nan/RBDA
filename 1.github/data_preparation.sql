SELECT id, created_at, url, title, body
FROM(
SELECT id, created_at, url, title, body
   , ROW_NUMBER() OVER (PARTITION BY SUBSTR(body, 80, 120) ORDER BY url) as count_body_beg
FROM(
SELECT id, created_at, url, title, body
   , ROW_NUMBER() OVER (PARTITION BY SUBSTR(body, 40, 80) ORDER BY url) as count_body_beg
FROM(
SELECT id, created_at, url, title, body
 , ROW_NUMBER() OVER (PARTITION BY SUBSTR(body, 0, 40) ORDER BY url) as count_body_beg
FROM(
    SELECT DISTINCT 
      id
    , created_at
    , url
    , REGEXP_REPLACE(title, r"\s{2,}", ' ') as title
    , REGEXP_REPLACE(body, r"\s{2,}", ' ') as body
    , ROW_NUMBER() OVER (PARTITION BY SUBSTR(title, 0, 22) ORDER BY url) as count_title_beg
    FROM(
        SELECT
            id
          , created_at
          , JSON_EXTRACT(payload, '$.issue.html_url') as url
          , LOWER(TRIM(REGEXP_REPLACE(JSON_EXTRACT(payload, '$.issue.title'), r"\\n|\(|\)|\[|\]|#|\*|`|\"", ' '))) as title
          , LOWER(TRIM(REGEXP_REPLACE(JSON_EXTRACT(payload, '$.issue.body'), r"\\n|\(|\)|\[|\]|#|\*|`|\"", ' '))) as body
        FROM `githubarchive.day.2024*`
        WHERE  
              _TABLE_SUFFIX BETWEEN '0101' and '1031'
          and type="IssuesEvent" 
          and JSON_EXTRACT(payload, '$.action') = "\"opened\"" 
        UNION ALL 
        SELECT
            id
          , created_at
          , JSON_EXTRACT(payload, '$.issue.html_url') as url
          , LOWER(TRIM(REGEXP_REPLACE(JSON_EXTRACT(payload, '$.issue.title'), r"\\n|\(|\)|\[|\]|#|\*|`|\"", ' '))) as title
          , LOWER(TRIM(REGEXP_REPLACE(JSON_EXTRACT(payload, '$.issue.body'), r"\\n|\(|\)|\[|\]|#|\*|`|\"", ' '))) as body
        FROM `githubarchive.day.2023*`
        WHERE
              _TABLE_SUFFIX BETWEEN '0101' and '1231'
          and type="IssuesEvent" 
          and JSON_EXTRACT(payload, '$.action') = "\"opened\""
    ) as tbl
    WHERE 
          ARRAY_LENGTH(SPLIT(body, ' ')) >= 10
      and ARRAY_LENGTH(SPLIT(title, ' ')) >= 5
      and LENGTH(title) <= 400
      and LENGTH(body) <= 2000
) tbl2
WHERE count_title_beg = 1
)tbl3
WHERE count_body_beg = 1
)tbl4
WHERE count_body_beg = 1
)tbl5
WHERE count_body_beg = 1