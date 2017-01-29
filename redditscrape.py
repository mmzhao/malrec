import datetime
import json
import praw
import time

pw = 'mrfrog273'

def scrapeRedditPosts(num_posts, outfile=None):
    r = praw.Reddit(client_id='XwuetZy7Y3LMIQ',
                client_secret='zCB2xjAEBB4Mufq6ObxURmzGMiU',
                user_agent='malrec',
                username='Ploebian',
                password='mrfrog273')
    ranime = r.subreddit('anime')
    start_time = time.time()
    pids = []
    while len(pids) < num_posts:
        params = {}
        if len(params):
            params = {"before" : "t3_{}".format(pids[-1]), "count" : len(pids)}
            posts = list(ranime.top('all', limit=num_posts, after="t3_{}".format(pids[-1])))
        else:
            posts = list(ranime.top('all', limit=num_posts))
        # posts = list(ranime.top('all', limit=num_posts, params=params))
        print "time spent finding posts:", time.time() - start_time
        pids += [p.id for p in posts]
        print len(pids)
        print len(set(pids))
        print pids[-1]
    pobj = {}
    pobj['posts'] = pids
    with open(outfile, 'w') as f:
        json.dump(pobj, f, indent=2)

def getRedditPosts(infile="None"):
    with open(infile, 'r') as f:
        posts = json.load(f)['posts']
    return posts

def scrapeRedditUsers(posts, start, end, infile=None, outfile=None):
    usernames = set()
    if infile:
        with open(infile, 'r') as f:
            usernames = set(json.load(f)['usernames'])
    r = praw.Reddit(client_id='XwuetZy7Y3LMIQ',
                client_secret='zCB2xjAEBB4Mufq6ObxURmzGMiU',
                user_agent='malrec',
                username='Ploebian',
                password=pw)
    ranime = r.subreddit('anime')
    # posts = list(ranime.hot(limit=100))

    tries = 3
    start_time = time.time()
    # posts = list(ranime.top('all', limit=num_posts))
    # print "time spent finding posts:", time.time() - start_time
    # pids = [p.id for p in posts]
    for i in range(start, min(len(posts), end)):
        if i%10 == 0 and i != start:
            print "[INFO] scraped {} posts, time spent: {}".format(i, time.time() - start_time)
            print "[INFO] usernames scraped:", len(usernames)
            uobj = {}
            uobj['usernames'] = list(usernames)
            with open(outfile, 'w') as f:
                json.dump(uobj, f, indent=2)
            print '[INFO] saved post ids'
        for _ in range(tries):
            try:
                p = r.submission(posts[i])
                print "post: {}, {}". format(p, i)
                p.comments.replace_more(limit=None, threshold=5)
                ranime_comments = p.comments.list()
                flairs = [c.author_flair_text for c in ranime_comments if c.author_flair_text]
                if p.author_flair_text:
                    flairs += [p.author_flair_text]
                for f in flairs:
                    preindex = f.find('myanimelist.net/profile/')
                    if preindex >= 0:
                        preindex += 24
                        postindex = f.find('?', preindex)
                        if postindex >= 0:
                            usernames.add(f[preindex:postindex])
                        else:
                            usernames.add(f[preindex:])
                    preindex = f.find('myanimelist.net/animelist/')
                    if preindex >= 0:
                        preindex += 26
                        postindex = f.find('?', preindex)
                        if postindex >= 0:
                            usernames.add(f[preindex:postindex])
                        else:
                            usernames.add(f[preindex:])
                    # print usernames
                print "number comments: {}, scraped users: {}".format(len(ranime_comments), len(usernames))
                break
            except KeyboardInterrupt:
                print "[ERROR] forced termination"
                print "[INFO] scraped {} posts, time spent: {}".format(i, time.time() - start_time)
                print "[INFO] usernames scraped:", len(usernames)
                uobj = {}
                uobj['usernames'] = list(usernames)
                with open(outfile, 'w') as f:
                    # json.dump(uobj, f)
                    json.dump(uobj, f, indent=2)
                print '[INFO] saved post ids'
                return
            except Exception as e:
                print '[ERROR] ', posts[i]
                print '[ERROR] ', e
    print "[INFO] scraped {} posts, time spent: {}".format(i, time.time() - start_time)
    print "[INFO] usernames scraped:", len(usernames)
    uobj = {}
    uobj['usernames'] = list(usernames)
    with open(outfile, 'w') as f:
        # json.dump(uobj, f)
        json.dump(uobj, f, indent=2)
    print '[INFO] saved post ids'

def addUsersFromTextFile(textfile=None, infile=None, outfile=None):
    usernames = set()
    if infile:
        with open(infile, 'r') as f:
            usernames = set(json.load(f)['usernames'])
    with open(textfile, 'r') as f:
        users = f.readlines()
    new_usernames = [u.split()[2] for u in users]
    usernames |= set(new_usernames)
    uobj = {}
    uobj['usernames'] = list(usernames)
    with open(outfile, 'w') as f:
        json.dump(uobj, f, indent=2)

def scrapeRedditPostsCloudSearch(start=None, end=None, infile=None, outfile=None):
    r = praw.Reddit(client_id='XwuetZy7Y3LMIQ',
                client_secret='zCB2xjAEBB4Mufq6ObxURmzGMiU',
                user_agent='malrec',
                username='Ploebian',
                password=pw)
    subreddit = 'anime'
    ranime = r.subreddit(subreddit)

    if end == None:
        end = datetime.datetime.now()
    if start == None:
        start = datetime.datetime.now() - datetime.timedelta(days=4*365)

    delta_days = 3

    upper = end
    lower = upper - datetime.timedelta(days=delta_days)
    pids = []
    if infile:
        with open(infile, 'r') as f:
            pids = json.load(f)['posts']

    missed_posts_ranges = []

    start_time = time.time()
    while upper > start:
        try:
            query = 'timestamp:%d..%d' % (int(lower.strftime("%s")), int(upper.strftime("%s")))
            posts = list(ranime.search(query, sort='new', limit=1000, syntax='cloudsearch'))
            if len(posts) >= 1000:
                print '[WARNING] posts may have been missed as this time range had >=1000 posts'
                missed_posts_ranges += [query]
            pids += [p.id for p in posts]
            print '[INFO] scraped {} posts from {} to {}'.format(len(posts), lower.strftime('%Y-%m-%d'), upper.strftime('%Y-%m-%d'))
            print '[INFO] total posts scraped: {}, days scraped: {}, time spent: {}'.format(len(pids), (end - lower).days, time.time() - start_time)
            upper = lower
            lower = upper - datetime.timedelta(days=delta_days)
        except KeyboardInterrupt:
            print "[ERROR] forced termination"
            break
        except Exception as e:
            print '[ERROR] ', e
    pobj = {}
    pobj['posts'] = pids
    with open(outfile, 'w') as f:
        json.dump(pobj, f, indent=2)
    print '[INFO] saved post ids'



# scrapeRedditPosts(10000, "posts_reddit.json")

# posts = getRedditPosts("posts_reddit.json")
# scrapeRedditUsers(posts, 8, 1000, "users_reddit.json", "users_reddit.json")


# addUsersFromTextFile(textfile="user_redditanimelist.txt", infile="users_reddit.json", outfile="users_ral.json")

# scrapeRedditPostsCloudSearch(outfile="posts_all_reddit.json")
# end = datetime.datetime.now() - datetime.timedelta(days=288) #2016-05-10
# print end.strftime('%Y-%m-%d')
# scrapeRedditPostsCloudSearch(end=end, infile="posts_all_reddit.json", outfile="posts_all_reddit.json")

posts = getRedditPosts("posts_all_reddit.json")
scrapeRedditUsers(posts, 0, 150000, infile="users_all_reddit.json", outfile="users_all_reddit.json")








