# auth.py - Zerodha Kite Connect Authentication System
import os
import datetime
from kiteconnect import KiteConnect
import config

TOKEN_FILE = "access_token.txt"

def get_kite_client():
    """
    Zerodha KiteConnect கணக்கை இணைத்து Access Token-ஐப் பெறுகிறது.
    தினமும் ஒருமுறை லாகின் செய்தால் போதுமானது.
    """
    kite = KiteConnect(api_key=config.API_KEY)
    
    # இன்றைக்கு ஏற்கனவே Access Token சேமிக்கப்பட்டுள்ளதா என்று பார்க்கிறது
    if os.path.exists(TOKEN_FILE):
        file_mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(TOKEN_FILE))
        if file_mod_time.date() == datetime.date.today():
            with open(TOKEN_FILE, "r") as f:
                access_token = f.read().strip()
            kite.set_access_token(access_token)
            print("[SUCCESS] இன்றைக்கான Access Token வெற்றிகரமாக ஏற்றப்பட்டது.")
            return kite

    # புதிய லாகின் தேவைப்பட்டால்
    print("\n--- ZERODHA LOGIN REQUIRED ---")
    print(f"1. இந்த லிங்கை பிரவுசரில் திறக்கவும்: {kite.login_url()}")
    print("2. லாகின் செய்த பிறகு முகவரி பட்டியில் (URL Bar) வரும் 'request_token' மதிப்பை காப்பி செய்யவும்.")
    
    request_token = input("3. 'request_token'-ஐ இங்கே பேஸ்ட் செய்யவும்: ").strip()
    
    try:
        data = kite.generate_session(request_token, api_secret=config.API_SECRET)
        access_token = data["access_token"]
        
        # இன்றைக்குப் பயன்படுத்தச் சேமிக்கிறது
        with open(TOKEN_FILE, "w") as f:
            f.write(access_token)
            
        kite.set_access_token(access_token)
        print("[SUCCESS] புதிய Session உருவாக்கப்பட்டு Access Token சேமிக்கப்பட்டது!")
        return kite
    except Exception as e:
        print(f"[ERROR] லாகின் செய்வதில் தோல்வி: {e}")
        return None

if __name__ == "__main__":
    print("Zerodha Kite இணைப்பைச் சோதிக்கிறது...")
    kite_session = get_kite_client()
    if kite_session:
        try:
            profile = kite_session.profile()
            print(f"[SUCCESS] இணைக்கப்பட்ட கணக்கு: {profile['user_name']} ({profile['user_id']})")
        except Exception as err:
            print(f"[ERROR] Profile பெற முடியவில்லை: {err}")