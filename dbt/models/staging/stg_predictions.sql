with source as (
    select * from {{ source('postgres', 'predictions') }}
),

renamed as (
    select
        request_id,
        status,
        result_json,
        updated_at::timestamp as processed_at
    from source
    where status = 'COMPLETED'
)

select * from renamed
