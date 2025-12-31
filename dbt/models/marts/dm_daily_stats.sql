with interactions as (
    select * from {{ ref('stg_interactions') }}
),

predictions as (
    select * from {{ ref('stg_predictions') }}
),

daily_active_users as (
    select
        date(created_at) as report_date,
        count(distinct user_id) as dau
    from interactions
    group by 1
),

daily_predictions as (
    select
        date(processed_at) as report_date,
        count(*) as total_predictions
    from predictions
    group by 1
),

final as (
    select
        coalesce(d.report_date, p.report_date) as report_date,
        coalesce(d.dau, 0) as dau,
        coalesce(p.total_predictions, 0) as total_predictions
    from daily_active_users d
    full outer join daily_predictions p on d.report_date = p.report_date
)

select * from final
order by report_date desc
