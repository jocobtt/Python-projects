import spotipy
from spotipy.oauth2 import SpotifyClientCredentials  # to access authorized spotify data

sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(client_id='8ed796f6fcfc404f89fd93fb6ffbe51c',
client_secret='ff5c3221f1b64fbf97dda2a376fe7284'))

results = sp.search(q='efence', limit=25)
for i, t in enumerate(results['tracks']['items']):
    print(' ', i, t['name'])



'''
client_id = "8ed796f6fcfc404f89fd93fb6ffbe51c"
client_secret = "ff5c3221f1b64fbf97dda2a376fe7284"

client_credentials_manager = SpotifyClientCredentials(client_id=client_id,
client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager) # spotify object to access API
# to get an artists id run - GET https://api.spotify.com/v1/artists/{id}
# results = sp.artist_top_tracks(artist_id=)  # my chosen artist
efence_uri = "Efence"



result = sp.search(efence_uri)  #search query

result['tracks']['items'][0]['artists']



birdy_uri = 'spotify:artist:2WX2uTcsvV5OnS0inACecP'


results = sp.artist_albums(birdy_uri, album_type='album')
albums = results['items']
while results['next']:
    results = spotify.next(results)
    albums.extend(results['items'])

for album in albums:
    print(album['name'])
'''
