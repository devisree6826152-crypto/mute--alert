import unittest
import json
import os
import sys
import uuid

# Ensure root folder is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app
from database.database import init_db, get_db_connection

class SOSEmergencySystemTestCase(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()
        init_db()

    def test_01_contacts_crud_flow(self):
        """Test adding, updating, fetching, and deleting emergency contacts."""
        # 1. Add contact
        resp = self.client.post('/api/contacts', data=json.dumps({
            'name': 'Jane Doe',
            'phone': '+15559876543',
            'relationship': 'Sister'
        }), content_type='application/json')
        
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertTrue(data['success'])
        contact_id = data['inserted_id']

        # 2. Fetch contacts
        get_resp = self.client.get('/api/contacts')
        self.assertEqual(get_resp.status_code, 200)
        get_data = json.loads(get_resp.data)
        self.assertTrue(get_data['success'])
        self.assertTrue(any(c['id'] == contact_id for c in get_data['contacts']))

        # 3. Update contact
        put_resp = self.client.put(f'/api/contacts/{contact_id}', data=json.dumps({
            'name': 'Jane Doe Updated',
            'phone': '+15559876543',
            'relationship': 'Guardian'
        }), content_type='application/json')
        self.assertEqual(put_resp.status_code, 200)

        # 4. Delete contact
        del_resp = self.client.delete(f'/api/contacts/{contact_id}')
        self.assertEqual(del_resp.status_code, 200)

    def test_02_sos_idempotency_and_event_flow(self):
        """Test SOS event creation, server-side idempotency check, and resolution."""
        sos_id = f"test_sos_{uuid.uuid4().hex}"

        payload = {
            "sos_id": sos_id,
            "emergency_type": "Medical Emergency",
            "latitude": 37.7749,
            "longitude": -122.4194,
            "location_mode": "gps",
            "accuracy": 10,
            "maps_url": "https://maps.google.com/?q=37.7749,-122.4194",
            "timestamp": "2026-09-17T12:00:00Z",
            "battery_status": "85% (Discharging)",
            "contacts": '[{"name":"Doc","phone":"555-0100"}]',
            "note": "Test SOS payload"
        }

        # First POST - create event
        resp1 = self.client.post('/api/sos', data=json.dumps(payload), content_type='application/json')
        self.assertEqual(resp1.status_code, 200)
        data1 = json.loads(resp1.data)
        self.assertTrue(data1['success'])
        self.assertFalse(data1['is_duplicate'])

        # Second POST with SAME sos_id - idempotency check must reject duplication
        resp2 = self.client.post('/api/sos', data=json.dumps(payload), content_type='application/json')
        self.assertEqual(resp2.status_code, 200)
        data2 = json.loads(resp2.data)
        self.assertTrue(data2['success'])
        self.assertTrue(data2['is_duplicate'])
        self.assertEqual(data2['event_id'], data1['event_id'])

        # Resolve event
        resolve_resp = self.client.put('/api/sos/resolve', data=json.dumps({'sos_id': sos_id}), content_type='application/json')
        self.assertEqual(resolve_resp.status_code, 200)
        res_data = json.loads(resolve_resp.data)
        self.assertTrue(res_data['success'])

if __name__ == '__main__':
    unittest.main()
