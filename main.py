from data import *
import time
warnings.filterwarnings('ignore')

if os.path.exists(save_path)==False:
    os.mkdir(save_path)
# 全量训练集
print("{}:Start".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
all_click_df = get_all_click_df(root, offline=False)
i2i_sim = itemcf_sim(all_click_df)

#给每个用户根据物品的协同过滤推荐文章
# 定义
print("{}:给每个用户根据物品的协同过滤推荐文章".format(time.strftime("%Y-%m-%d %H:%M:%S",
                                                   time.localtime())))
user_recall_items_dict = collections.defaultdict(dict)
# 获取 用户 - 文章 - 点击时间的字典
print("{}:获取用户 - 文章 - 点击时间的字典".format(time.strftime("%Y-%m-%d %H:%M:%S",
                                                    time.localtime()) ))
user_item_time_dict = get_user_item_time(all_click_df)
# 去取文章相似度
print("{}:去取文章相似度".format(time.strftime("%Y-%m-%d %H:%M:%S",
                                        time.localtime()) ))
i2i_sim = pickle.load(open(save_path + 'itemcf_i2i_sim.pkl', 'rb'))
# 相似文章的数量
sim_item_topk = 10
# 召回文章数量
recall_item_num = 10

# 用户热度补全
item_topk_click = get_item_topk_click(all_click_df, k=50)

for user in tqdm(all_click_df['user_id'].unique()):
    user_recall_items_dict[user] = item_based_recommend(user, user_item_time_dict,
                                                        i2i_sim,
                                                        sim_item_topk,
                                                        recall_item_num,
                                                        item_topk_click)

#召回字典转换成df
# 将字典的形式转换成df
user_item_score_list = []

for user, items in tqdm(user_recall_items_dict.items()):
    for item, score in items:
        user_item_score_list.append([user, item, score])

recall_df = pd.DataFrame(user_item_score_list, columns=['user_id',
                                                        'click_article_id',
                                                        'pred_score'])
#生成提交文件
# 获取测试集
tst_click = pd.read_csv(save_path + 'testA_click_log.csv')
tst_users = tst_click['user_id'].unique()

# 从所有的召回数据中将测试集中的用户选出来
tst_recall = recall_df[recall_df['user_id'].isin(tst_users)]

# 生成提交文件
submit(tst_recall, topk=5, model_name='itemcf_baseline')

