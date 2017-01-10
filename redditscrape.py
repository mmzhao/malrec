import json
import praw
import time

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
                password='mrfrog273')
    ranime = r.subreddit('anime')
    # posts = list(ranime.hot(limit=100))
    start_time = time.time()
    # posts = list(ranime.top('all', limit=num_posts))
    # print "time spent finding posts:", time.time() - start_time
    # pids = [p.id for p in posts]
    for i in range(start, min(len(posts), end)):
        p = r.submission(posts[i])
        print "post:", p, i
        p.comments.replace_more(limit=1000, threshold=10)
        ranime_comments = p.comments.list()
        print "number comments:", len(ranime_comments)
        flairs = [c.author_flair_text for c in ranime_comments if c.author_flair_text]
        for f in flairs:
            preindex = f.find('myanimelist.net/profile/')
            if preindex >= 0:
                preindex += 24
                postindex = f.find('?', preindex)
                if postindex >= 0:
                    usernames.add(f[preindex:postindex])
                else:
                    usernames.add(f[preindex:])
            # print usernames
        print "time spent:", time.time() - start_time
        print "usernames scraped:", len(usernames)
        uobj = {}
        uobj['usernames'] = list(usernames)
        with open(outfile, 'w') as f:
            # json.dump(uobj, f)
            json.dump(uobj, f, indent=2)


# scrapeRedditPosts(10000, "posts_reddit.json")

# posts = getRedditPosts("posts_reddit.json")
# scrapeRedditUsers(posts, 8, 1000, "users_reddit.json", "users_reddit.json")














