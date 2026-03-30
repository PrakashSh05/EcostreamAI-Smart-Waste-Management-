def get_city_from_coords(lat: float, lng: float) -> str:
    # Bangalore (lat 12.8-13.2, lng 77.4-77.8)
    if 12.8 <= lat <= 13.2 and 77.4 <= lng <= 77.8:
        return "Bangalore"
    # Mumbai approx (lat 18.8-19.3, lng 72.7-73.0)
    elif 18.8 <= lat <= 19.3 and 72.7 <= lng <= 73.0:
        return "Mumbai"
    # Delhi approx (lat 28.4-28.9, lng 76.8-77.3)
    elif 28.4 <= lat <= 28.9 and 76.8 <= lng <= 77.3:
        return "Delhi"
    # Chennai approx (lat 12.9-13.3, lng 80.1-80.3)
    elif 12.9 <= lat <= 13.3 and 80.1 <= lng <= 80.3:
        return "Chennai"
    
    return "India"
