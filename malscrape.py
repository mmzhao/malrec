from bs4 import BeautifulSoup as Soup
import collections
from collections import OrderedDict
import cProfile
import json
import httplib
import re
import requests
import sys
import time
import urllib2

def scrapeId2AnimeDict(num_anime, outfile='animes_sorted.json'):
    id2anime = OrderedDict()
    for i in range(num_anime//50):
        html = urllib2.urlopen("https://myanimelist.net/topanime.php?type=bypopularity&limit={}".format(50 * i)).read()
        # with open('temp.txt', 'w') as f:
            #   f.write(html)
        p = re.compile('href=\"https://myanimelist.net/anime/\d+/[^/\"]+\" id=')
        animes = p.findall(html)
        animes = [animes[2 * i] for i in range(len(animes)//2)]
        for anime in animes:
            fields = anime.split('\"')[1].split('/')
            id2anime[int(fields[4])] = fields[5]
        # print len(id2anime.keys())
        # time.sleep(1)
    # with open('animes.json', 'w') as f:
    #     # json.dump(id2anime, f)
    #     json.dump(id2anime, f, indent=2)

    # id2anime = {int(k):v for k,v in id2anime.items()}
    with open(oufile, 'w') as f:
        # json.dump(id2anime, f)
        json.dump(OrderedDict(sorted(id2anime.items())), f, indent=2)

# isn't sorted when loaded from JSON
def getId2AnimeDictSorted(infile='animes_sorted.json'):
    with open(infile, 'r') as f:
        id2anime = json.load(f, object_pairs_hook=OrderedDict)
    id2anime = {int(k):v for k,v in id2anime.items()}
    return OrderedDict(sorted(id2anime.items()))

def getId2AnimeDictUnsorted(infile='animes_unsorted.json'):
    with open(infile, 'r') as f:
        id2anime = json.load(f, object_pairs_hook=OrderedDict)
    return id2anime

def scrapeClubs(num_clubs, outfile='clubs.json'):
    cid2memmbers = OrderedDict()
    # for i in range(1, 16):
    for i in range(1, num_clubs//50 + 1):
        html = urllib2.urlopen("https://myanimelist.net/clubs.php?sort=5&p={}".format(i)).read()
        # with open('temp.txt', 'w') as f:
        #   f.write(html)
        p = re.compile('https://myanimelist.net/clubs.php\?cid=\d+.+\n.+\n.+\n.+')
        clubs = p.findall(html)
        # clubs = [clubs[2 * i] for i in range(len(clubs)//2)]
        # print clubs, len(clubs)
        for club in clubs:
            cid = club.split("\"")[0].split('=')[1]
            members = int(club.split("\n")[-1].replace(',', ''))
            cid2memmbers[cid] = members
    with open(outfile, 'w') as f:
        # json.dump(cid2memmbers, f)
        json.dump(cid2memmbers, f, indent=2)

def getClubs(infile='clubs.json'):
    with open(infile, 'r') as f:
        clubs = json.load(f, object_pairs_hook=OrderedDict)
    return clubs

def scrapeUsers(clubs, num_clubs, outfile='users.json'):
    usernames = set()
    cids = clubs.keys()
    for i in range(min(num_clubs, len(cids))):
        print "scrape new club", cids[i]
        for j in range(clubs[cids[i]]//36):
        # for j in range(1):
            html = urllib2.urlopen("https://myanimelist.net/clubs.php?action=view&t=members&id={0}&show={1}".format(cids[i], j*36)).read()
            # with open('temp.txt', 'w') as f:
            #   f.write(html)
            p = re.compile('<a href=\"/profile/.+?\"><')
            users = p.findall(html)
            # print users, len(users)
            # print users[0].split('/')[2].split("\"")[0]
            for user in users:
                usernames.add(user.split('/')[2].split("\"")[0])
            print len(usernames)
        uobj = {}
        uobj['usernames'] = list(usernames)
        with open(outfile, 'w') as f:
            # json.dump(uobj, f)
            json.dump(uobj, f, indent=2)
    # usernames = list(usernames)
    # uobj = {}
    # uobj['usernames'] = list(usernames)
    # with open('users.json', 'w') as f:
    #     # json.dump(uobj, f)
    #     json.dump(uobj, f, indent=2)

def scrapeAddUsers(clubs, start, end, infile='users.json', outfile='users.json'):
    with open(infile, 'r') as f:
        usernames = set(json.load(f)['usernames'])
    cids = clubs.keys()
    for i in range(start, min(end, len(cids))):
        print "scrape new club", cids[i]
        for j in range(clubs[cids[i]]//36):
        # for j in range(1):
            html = urllib2.urlopen("https://myanimelist.net/clubs.php?action=view&t=members&id={0}&show={1}".format(cids[i], j*36)).read()
            # with open('temp.txt', 'w') as f:
            #   f.write(html)
            p = re.compile('<a href=\"/profile/.+?\"><')
            users = p.findall(html)
            # print users, len(users)
            # print users[0].split('/')[2].split("\"")[0]
            for user in users:
                usernames.add(user.split('/')[2].split("\"")[0])
            print j, "out of", clubs[cids[i]]//36, ":", len(usernames)
        uobj = {}
        uobj['usernames'] = list(usernames)
        with open(outfile, 'w') as f:
            # json.dump(uobj, f)
            json.dump(uobj, f, indent=2)
        print "done scraping", cids[i]

def getUsers(infile='users.json'):
    with open(infile, 'r') as f:
        users = json.load(f)['usernames']
    return users

# def getUsersSorted():
#     with open('users_sorted.json', 'r') as f:
#         users = json.load(f)['usernames']
#     return users

# def getUsersUnsorted():
#     with open('users_unsorted.json', 'r') as f:
#         users = json.load(f)['usernames']
#     return users

def scrapeAnimelists(users, num_users, id2anime, outfile='user_animelists.json'):
    start = time.time()
    user2animelist = {}
    for i in range(min(num_users, len(users))):
        # html = urllib2.urlopen("https://myanimelist.net/animelist/{}?status=2".format(users[i])).read()
        html = urllib2.urlopen("https://myanimelist.net/malappinfo.php?u={}&status=all&type=anime".format(users[i])).read()
        # with open('temp.txt', 'w') as f:
        #   f.write(html)
        # p = re.compile('<a href=\"/anime/32998/.+\".+')
        # USING MALAPPINFO BYPASSES PRIVATE ANIMELISTS LEL
        # private = re.compile('Access to this list has been restricted by the owner.')
        # if len(private.findall(html)) > 0:
        #     print "private animelist:", users[i], i
        #     continue
        # p = re.compile('score.+?anime_id.+?\d+')
        p = re.compile('<series_animedb_id>\d+.+?<my_score>\d+.+?<my_status>2')
        scores = p.findall(html)
        if len(scores) == 0:
            print "empty animelist:", users[i], i
            continue
        # print scores, len(scores)
        anime2score = {}
        for score in scores:
            # print scores[0].replace('<', '>').split('>')[2]
            # print scores[0].replace('<', '>').split('>')[-5]
            fields = score.replace('<', '>').split('>')
            anime2score[int(fields[2])] = int(fields[-5])
            # print int(score.split(':')[1].split(',')[0])
            # print int(score.split(':')[-1])
        anime2score = OrderedDict(sorted(anime2score.items()))
        # print len(scores)
        user2animelist[users[i]] = anime2score
        if (i + 1)%100 == 0:
            with open(outfile, 'w') as f:
                json.dump(user2animelist, f)
            print "time spent:", time.time() - start
            print "scraped {0} users, {1} non-empty animelists, ending with: {2}".format(i + 1, len(user2animelist), users[i])
    with open(outfile, 'w') as f:
        json.dump(user2animelist, f)
        # json.dump(user2animelist, f, indent=2)

def scrapeAddAnimelists(users, start, end, id2anime, infile='user_animelists.json', outfile='user_animelists.json'):
    with open(infile, 'r') as f:
        user2animelist = json.load(f)
    start_time = time.time()
    for i in range(start, min(end, len(users))):
        # html = urllib2.urlopen("https://myanimelist.net/animelist/{}?status=2".format(users[i])).read()
        html = urllib2.urlopen("https://myanimelist.net/malappinfo.php?u={}&status=all&type=anime".format(users[i])).read()
        # with open('temp.txt', 'w') as f:
        #   f.write(html)
        # p = re.compile('<a href=\"/anime/32998/.+\".+')
        # USING MALAPPINFO BYPASSES PRIVATE ANIMELISTS LEL
        # private = re.compile('Access to this list has been restricted by the owner.')
        # if len(private.findall(html)) > 0:
        #     print "private animelist:", users[i], i
        #     continue
        # p = re.compile('score.+?anime_id.+?\d+')
        p = re.compile('<series_animedb_id>\d+.+?<my_score>\d+.+?<my_status>2')
        scores = p.findall(html)
        if len(scores) == 0:
            print "empty animelist:", users[i], i
            continue
        # print scores, len(scores)
        anime2score = {}
        for score in scores:
            # print scores[0].replace('<', '>').split('>')[2]
            # print scores[0].replace('<', '>').split('>')[-5]
            fields = score.replace('<', '>').split('>')
            anime2score[int(fields[2])] = int(fields[-5])
            # print int(score.split(':')[1].split(',')[0])
            # print int(score.split(':')[-1])
        anime2score = OrderedDict(sorted(anime2score.items()))
        # print len(scores)
        user2animelist[users[i]] = anime2score
        if (i + 1)%100 == 0:
            with open(outfile, 'w') as f:
                json.dump(user2animelist, f)
            print "time spent:", time.time() - start_time
            print "scraped {0} users, {1} non-empty animelists, ending with: {2}".format(i + 1, len(user2animelist), users[i])
    with open(outfile, 'w') as f:
        json.dump(user2animelist, f)
        # json.dump(user2animelist, f, indent=2)

def scrapeAnimelistsPersistant(users, num_users, id2anime, outfile='user_animelists.json'):
    user2animelist = {}
    conn = httplib.HTTPSConnection("myanimelist.net")
    start = time.time()
    for i in range(min(num_users, len(users))):
        conn.request("GET", "/malappinfo.php?u={}&status=all&type=anime".format(users[i]))
        res = conn.getresponse()
        if res.status != 200:
            print "failed request on user:", users[i]
            print "status: {0}, reason: {1}".format(res.status, res.reason)
            continue
        xml = res.read()
        p = re.compile('<series_animedb_id>\d+.+?<my_score>\d+.+?<my_status>2')
        scores = p.findall(html)
        if len(scores) == 0:
            print "empty animelist:", users[i], i
            continue
        # print scores, len(scores)
        anime2score = {}
        for score in scores:
            # print scores[0].replace('<', '>').split('>')[2]
            # print scores[0].replace('<', '>').split('>')[-5]
            fields = score.replace('<', '>').split('>')
            anime2score[int(fields[2])] = int(fields[-5])
            # print int(score.split(':')[1].split(',')[0])
            # print int(score.split(':')[-1])
        anime2score = OrderedDict(sorted(anime2score.items()))
        # print len(scores)
        user2animelist[users[i]] = anime2score
        if (i + 1)%100 == 0:
            with open(outfile, 'w') as f:
                json.dump(user2animelist, f)
            print "time spent:", time.time() - start
            print "scraped {0} users, {1} non-empty animelists, ending with: {2}".format(i + 1, len(user2animelist), users[i])
    with open(outfile, 'w') as f:
        json.dump(user2animelist, f)
    conn.close()
        # json.dump(user2animelist, f, indent=2)

def scrapeAddAnimelistsPersistant(users, start, end, id2anime, infile='user_animelists.json', outfile='user_animelists.json'):
    with open(infile, 'r') as f:
        user2animelist = json.load(f)
    conn = httplib.HTTPSConnection("myanimelist.net")
    start_time = time.time()
    for i in range(start, min(end, len(users))):
        conn.request("GET", "/malappinfo.php?u={}&status=all&type=anime".format(users[i]))
        res = conn.getresponse()
        if res.status != 200:
            print "failed request on user:", users[i], i
            print "status: {0}, reason: {1}".format(res.status, res.reason)
            print "read: {}".format(res.read())
            continue
        xml = res.read()
        print users[i], i
        # print html
        p = re.compile('<series_animedb_id>\d+.+?<my_score>\d+.+?<my_status>2')
        scores = p.findall(xml)
        if len(scores) == 0:
            print "empty animelist:", users[i], i
            continue
        # print scores, len(scores)
        anime2score = {}
        for score in scores:
            # print scores[0].replace('<', '>').split('>')[2]
            # print scores[0].replace('<', '>').split('>')[-5]
            fields = score.replace('<', '>').split('>')
            anime2score[int(fields[2])] = int(fields[-5])
            # print int(score.split(':')[1].split(',')[0])
            # print int(score.split(':')[-1])
        anime2score = OrderedDict(sorted(anime2score.items()))
        # print len(scores)
        user2animelist[users[i]] = anime2score
        if (i + 1)%100 == 0:
            with open(outfile, 'w') as f:
                json.dump(user2animelist, f)
            print "time spent:", time.time() - start_time
            print "scraped {0} users, {1} non-empty animelists, ending with: {2}".format(i + 1, len(user2animelist), users[i])
    with open(outfile, 'w') as f:
        json.dump(user2animelist, f)
        # json.dump(user2animelist, f, indent=2)
    conn.close()

def scrapeAnimelistsSoup(users, num_users, id2anime, outfile='user_animelists.json'):
    user2animelist = {}
    conn = httplib.HTTPSConnection("myanimelist.net")
    start = time.time()
    for i in range(min(num_users, len(users))):
        try:
            conn.request("GET", "/malappinfo.php?u={}&status=all&type=anime".format(users[i]))
            res = conn.getresponse()
            if res.status != 200:
                print "failed request on user:", users[i], i
                print "status: {0}, reason: {1}".format(res.status, res.reason)
                continue
            xml = res.read()
            soup = Soup(xml, 'lxml-xml')
            entries = soup.findAll('anime')
            anime2score = {}
            for entry in entries:
                my_status = int(entry.my_status.contents[0])
                if my_status != 2:
                    continue
                anime_id = int(entry.series_animedb_id.contents[0])
                my_score = int(entry.my_score.contents[0])
                anime2score[anime_id] = my_score
            anime2score = OrderedDict(sorted(anime2score.items()))
            if len(anime2score) == 0:
                print "empty animelist:", users[i], i
                continue
            user2animelist[users[i]] = anime2score
            if (i + 1)%100 == 0:
                with open(outfile, 'w') as f:
                    json.dump(user2animelist, f)
                print "time spent:", time.time() - start
                print "scraped {0} users, {1} non-empty animelists, ending with: {2}".format(i + 1, len(user2animelist), users[i])
        except KeyboardInterrupt:
            print "forced termination"
            return
        except:
            print "exception on user:", users[i], i
    with open(outfile, 'w') as f:
        json.dump(user2animelist, f)
    conn.close()

def scrapeAddAnimelistsSoup(users, start, end, id2anime, infile='user_animelists.json', outfile='user_animelists.json'):
    with open(infile, 'r') as f:
        user2animelist = json.load(f)
    conn = httplib.HTTPSConnection("myanimelist.net")
    start_time = time.time()
    for i in range(start, min(end, len(users))):
        try:
            conn.request("GET", "/malappinfo.php?u={}&status=all&type=anime".format(users[i]))
            res = conn.getresponse()
            if res.status != 200:
                print "failed request on user:", users[i], i
                print "status: {0}, reason: {1}".format(res.status, res.reason)
                print "read: {}".format(res.read())
                continue
            xml = res.read()
            soup = Soup(xml, 'lxml-xml')
            entries = soup.findAll('anime')
            anime2score = {}
            for entry in entries:
                my_status = int(entry.my_status.contents[0])
                if my_status != 2:
                    continue
                anime_id = int(entry.series_animedb_id.contents[0])
                my_score = int(entry.my_score.contents[0])
                anime2score[anime_id] = my_score
            anime2score = OrderedDict(sorted(anime2score.items()))
            if len(anime2score) == 0:
                print "empty animelist:", users[i], i
                continue
            user2animelist[users[i]] = anime2score
            if (i + 1)%100 == 0:
                with open(outfile, 'w') as f:
                    json.dump(user2animelist, f)
                print "time spent:", time.time() - start_time
                print "scraped {0} users, {1} non-empty animelists, ending with: {2}".format(i + 1, len(user2animelist), users[i])
        except KeyboardInterrupt:
            print "forced termination"
            return
        except:
            print "exception on user:", users[i], i
    with open(outfile, 'w') as f:
        json.dump(user2animelist, f)
    conn.close()

def scrapeAnimelistsNew(users, num_users, id2anime, outfile='user_animelists.json', tries=3):
    user2animelist = {}
    s = requests.Session()
    start = time.time()
    for i in range(min(num_users, len(users))):
        for _ in range(tries):
            try:
                res = s.get("https://myanimelist.net/malappinfo.php?u={}&status=all&type=anime".format(users[i]))
                if res.status_code != 200:
                    raise Exception("failed request on user:", users[i], i, "status: {0}".format(res.status_code))
                xml = res.text
                soup = Soup(xml, 'lxml-xml')
                entries = soup.findAll('anime')
                anime2score = {}
                for entry in entries:
                    my_status = int(entry.my_status.contents[0])
                    if my_status != 2:
                        continue
                    anime_id = int(entry.series_animedb_id.contents[0])
                    my_score = int(entry.my_score.contents[0])
                    anime2score[anime_id] = my_score
                anime2score = OrderedDict(sorted(anime2score.items()))
                if len(anime2score) == 0:
                    print "empty animelist:", users[i], i
                    break
                user2animelist[users[i]] = anime2score
                if (i + 1)%100 == 0:
                    with open(outfile, 'w') as f:
                        json.dump(user2animelist, f)
                    print "time spent:", time.time() - start_time
                    print "scraped {0} users, {1} non-empty animelists, ending with: {2}".format(i + 1, len(user2animelist), users[i])
                break
            except KeyboardInterrupt:
                print "forced termination"
                return
            except Exception as e:
                print "exception on user:", users[i], i
                print e
                time.sleep(1)
                continue
    with open(outfile, 'w') as f:
        json.dump(user2animelist, f)

def scrapeAddAnimelistsNew(users, start, end, id2anime, infile='user_animelists.json', outfile='user_animelists.json', tries=3):
    with open(infile, 'r') as f:
        user2animelist = json.load(f)
    s = requests.Session()
    start_time = time.time()
    for i in range(start, min(end, len(users))):
        for _ in range(tries):
            print _
            time.sleep(.5)
            try:
                res = s.get("https://myanimelist.net/malappinfo.php?u={}&status=all&type=anime".format(users[i]))
                if res.status_code != 200:
                    raise Exception("failed request on user:", users[i], i, "status: {0}".format(res.status_code))
                xml = res.text
                soup = Soup(xml, 'lxml-xml')
                entries = soup.findAll('anime')
                anime2score = {}
                for entry in entries:
                    my_status = int(entry.my_status.contents[0])
                    if my_status != 2:
                        continue
                    anime_id = int(entry.series_animedb_id.contents[0])
                    my_score = int(entry.my_score.contents[0])
                    anime2score[anime_id] = my_score
                anime2score = OrderedDict(sorted(anime2score.items()))
                if len(anime2score) == 0:
                    print "empty animelist:", users[i], i
                    break
                user2animelist[users[i]] = anime2score
                if (i + 1)%100 == 0:
                    with open(outfile, 'w') as f:
                        json.dump(user2animelist, f)
                    print "time spent:", time.time() - start_time
                    print "scraped {0} users, {1} non-empty animelists, ending with: {2}".format(i + 1, len(user2animelist), users[i])
                break
            except KeyboardInterrupt:
                print "forced termination"
                return
            except Exception as e:
                print "exception on user:", users[i], i
                print e
                time.sleep(1)
                continue
    with open(outfile, 'w') as f:
        json.dump(user2animelist, f)

def getAnimelists(infile='user_animelists.json'):
    with open(infile, 'r') as f:
        user2animelist = json.load(f)
    return user2animelist

def makeArrays():
    user_anime_sorted = anime2score.keys()
    user_anime_sorted.sort()
    all_anime_sorted = id2anime.keys()
    full_scores = []
    user_index = 0
    all_index = 0
    scores_1000 = 0
    while user_index < len(user_anime_sorted) and all_index < len(all_anime_sorted):
        # print user_index, all_index, user_anime_sorted[user_index], all_anime_sorted[all_index]
        if user_anime_sorted[user_index] == all_anime_sorted[all_index]:
            full_scores += [anime2score[user_anime_sorted[user_index]]]
            user_index += 1
            all_index += 1
            scores_1000 += 1
        elif user_anime_sorted[user_index] not in all_anime_sorted:
            # print 'anime not in top 1000 most popular:', user_anime_sorted[user_index] #, id2anime[user_anime_sorted[user_index]]
            user_index += 1
        else: # user_anime_sorted[user_index] > all_anime_sorted[all_index]:
            full_scores += [0]
            all_index += 1
    full_scores += [0 for i in range(len(all_anime_sorted) - len(full_scores))]
    print 'scores in top 1000 most popular:', scores_1000



# scrapeId2AnimeDict(1000)
# scrapeClubs(100)

# clubs = getClubs()
# scrapeUsers(clubs, 3)
# scrapeAddUsers(clubs, 3, 98)

id2anime = getId2AnimeDictSorted()
users = getUsers('users_club.json')
# print len(users)

# scrapeAnimelistsNew(['Ploebian'], 1, id2anime, outfile="wat")
# scrapeAddAnimelistsNew(users, 13500, 100000, id2anime, infile='user_animelists_club_first_half.json', outfile='user_animelists_club_first_half.json')
# scrapeAddAnimelistsNew(users, 100200, 200000, id2anime, infile='user_animelists_club_10000.json', outfile='user_animelists_club_second_half.json')
cProfile.run('scrapeAddAnimelistsNew(users, 100100, 100200, id2anime, infile="user_animelists_club_10000.json", outfile="user_animelists_club_second_half.json")', sort='tottime')


















