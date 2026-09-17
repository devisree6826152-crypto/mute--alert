import sqlite3
import os
from datetime import datetime

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'signspeak.db')

def get_db_path():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        os.makedirs(base_dir, exist_ok=True)
        # Test write capability
        test_file = os.path.join(base_dir, '.write_test')
        with open(test_file, 'w') as f:
            f.write('1')
        os.remove(test_file)
        return DEFAULT_DB_PATH
    except Exception:
        tmp_dir = os.path.join('/tmp', 'signspeak_db')
        os.makedirs(tmp_dir, exist_ok=True)
        return os.path.join(tmp_dir, 'signspeak.db')

def get_db_connection():
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize the SQLite database with history and dataset tables."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # History Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recognized_text TEXT NOT NULL,
                confidence REAL NOT NULL,
                timestamp TEXT NOT NULL,
                sign_type TEXT DEFAULT 'sentence'
            )
        ''')
        
        # Dataset Metadata Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dataset_meta (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                class_name TEXT UNIQUE NOT NULL,
                sample_count INTEGER DEFAULT 0,
                last_updated TEXT
            )
        ''')
        
        # SOS Events Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sos_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sos_id TEXT UNIQUE NOT NULL,
                emergency_type TEXT NOT NULL,
                latitude REAL,
                longitude REAL,
                location_mode TEXT DEFAULT 'gps',
                accuracy REAL,
                maps_url TEXT,
                timestamp TEXT NOT NULL,
                battery_status TEXT,
                contacts TEXT,
                note TEXT,
                status TEXT DEFAULT 'ACTIVE',
                resolved_at TEXT,
                user_id TEXT DEFAULT 'default'
            )
        ''')
        
        # Emergency Contacts Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS emergency_contacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                phone TEXT NOT NULL,
                relationship TEXT DEFAULT 'Contact',
                created_at TEXT NOT NULL,
                user_id TEXT DEFAULT 'default'
            )
        ''')
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Warning: Database initialization error: {e}")

def add_sos_event(sos_data):
    """Insert a new real SOS event record with idempotency safety."""
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    
    sos_id = sos_data.get('sos_id')
    user_id = sos_data.get('user_id', 'default')
    
    # Idempotency check: if sos_id already exists, return existing record ID without duplicating
    cursor.execute('SELECT id, status FROM sos_events WHERE sos_id = ?', (sos_id,))
    existing = cursor.fetchone()
    if existing:
        conn.close()
        return {'id': existing['id'], 'sos_id': sos_id, 'is_duplicate': True, 'status': existing['status']}
        
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
        INSERT INTO sos_events (
            sos_id, emergency_type, latitude, longitude, location_mode,
            accuracy, maps_url, timestamp, battery_status, contacts, note, status, user_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'ACTIVE', ?)
    ''', (
        sos_id,
        sos_data.get('emergency_type', 'General Danger'),
        sos_data.get('latitude'),
        sos_data.get('longitude'),
        sos_data.get('location_mode', 'gps'),
        sos_data.get('accuracy'),
        sos_data.get('maps_url'),
        sos_data.get('timestamp', now_str),
        sos_data.get('battery_status', 'Battery status unavailable'),
        sos_data.get('contacts', '[]'),
        sos_data.get('note', ''),
        user_id
    ))
    
    conn.commit()
    inserted_id = cursor.lastrowid
    conn.close()
    return {'id': inserted_id, 'sos_id': sos_id, 'is_duplicate': False, 'status': 'ACTIVE'}

def get_sos_events(limit=50, user_id='default'):
    """Fetch SOS history events ordered by newest first."""
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM sos_events WHERE user_id = ? ORDER BY id DESC LIMIT ?', (user_id, limit))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def resolve_sos_event(sos_id):
    """Mark an active SOS event as RESOLVED."""
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
        UPDATE sos_events
        SET status = 'RESOLVED', resolved_at = ?
        WHERE sos_id = ? OR id = ?
    ''', (now_str, str(sos_id), str(sos_id)))
    conn.commit()
    affected = cursor.rowcount
    conn.close()
    return affected > 0

def add_contact(name, phone, relationship='Contact', user_id='default'):
    """Add a new real emergency contact."""
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
        INSERT INTO emergency_contacts (name, phone, relationship, created_at, user_id)
        VALUES (?, ?, ?, ?, ?)
    ''', (name.strip(), phone.strip(), relationship.strip() or 'Contact', now_str, user_id))
    conn.commit()
    inserted_id = cursor.lastrowid
    conn.close()
    return inserted_id

def update_contact(contact_id, name, phone, relationship='Contact'):
    """Update an existing emergency contact."""
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE emergency_contacts
        SET name = ?, phone = ?, relationship = ?
        WHERE id = ?
    ''', (name.strip(), phone.strip(), relationship.strip() or 'Contact', int(contact_id)))
    conn.commit()
    affected = cursor.rowcount
    conn.close()
    return affected > 0

def get_contacts(user_id='default'):
    """Retrieve saved emergency contacts for the user."""
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM emergency_contacts WHERE user_id = ? ORDER BY id ASC', (user_id,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def delete_contact(contact_id):
    """Delete an emergency contact by ID."""
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM emergency_contacts WHERE id = ?', (int(contact_id),))
    conn.commit()
    affected = cursor.rowcount
    conn.close()
    return affected > 0

def add_history_record(recognized_text, confidence, sign_type='sentence'):
    """Insert a new recognized sign or sentence into the history database."""
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
        INSERT INTO history (recognized_text, confidence, timestamp, sign_type)
        VALUES (?, ?, ?, ?)
    ''', (recognized_text, float(confidence), now_str, sign_type))
    conn.commit()
    inserted_id = cursor.lastrowid
    conn.close()
    return inserted_id

def get_history(limit=100):
    """Retrieve recognition history records ordered by newest first."""
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM history ORDER BY id DESC LIMIT ?', (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def delete_history_record(record_id):
    """Delete a specific history record by ID."""
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM history WHERE id = ?', (record_id,))
    conn.commit()
    conn.close()
    return True

def clear_history():
    """Clear all records from the history table."""
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM history')
    conn.commit()
    conn.close()
    return True

def update_class_sample_count(class_name, count):
    """Update or insert sample count metadata for a sign class."""
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
        INSERT INTO dataset_meta (class_name, sample_count, last_updated)
        VALUES (?, ?, ?)
        ON CONFLICT(class_name) DO UPDATE SET
            sample_count = ?,
            last_updated = ?
    ''', (class_name, count, now_str, count, now_str))
    conn.commit()
    conn.close()

def get_dataset_stats():
    """Fetch class-wise sample statistics."""
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT class_name, sample_count, last_updated FROM dataset_meta ORDER BY class_name ASC')
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

if __name__ == '__main__':
    init_db()
    print("Database initialized successfully at:", DB_PATH)

