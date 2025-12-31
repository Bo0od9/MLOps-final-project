CREATE TABLE IF NOT EXISTS predictions (
    request_id VARCHAR(50) PRIMARY KEY,
    user_id VARCHAR(50),
    status VARCHAR(20),
    result_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS interactions (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(50) NOT NULL,
    item_id INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_user_id ON interactions(user_id);
CREATE INDEX idx_created_at ON interactions(created_at);
