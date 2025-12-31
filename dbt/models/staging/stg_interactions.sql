with source as (
    select * from {{ source('postgres', 'interactions') }}
),

renamed as (
    select
        distinct
        user_id,
        item_id,
        created_at
    from source
)

select * from renamed
