import numpy as np
import scipy.special as sc_special
import pandas as pd
import matplotlib.pyplot as plt
import scipy.spatial

class cuckoo:
    def __init__(self, n, m, buildings, x_boundary, y_boundary, iter_num = 70, pa = 0.25, step = 2):
        """
        Input============
        n:鸟数
        m:下了的槽数
        x_boundary-> 1*2 array (min,max)
        pa: 丢弃概率
        step: 更新的移动距离
        """
        self.n = n  # 鸟的数目
        self.m = int(m)  # 基站个数
        self.pa = pa
        self.step = step
        self.iter_num = iter_num
        self.buildings = buildings  # 小区位置
        self.station = {}
        self.x_boundary = np.tile(x_boundary, (self.m,1)).reshape(self.m, 2)
        self.y_boundary = np.tile(y_boundary, (self.m,1)).reshape(self.m, 2)
        self.x_nests = np.empty((n, self.m))
        self.y_nests = np.empty((n, self.m))
        self.fitness = np.empty((1, n))    #   np.zeros((1, n))+100
        self.best_fit = 100
        self.best_station_x = np.zeros(1*self.m)
        self.best_station_y = np.zeros(1*self.m)

    def generate_nests(self):
        """
        Generate the nests' locations
        ---------------------------------------------------
        Output:
            generated nests' locations
        """
        for each_nest in range(self.n):
            x_s = self.x_boundary[:,0] + np.array([np.random.rand() for _ in range(self.m)]) * (self.x_boundary[:,1]-self.x_boundary[:,0])
            y_s = self.y_boundary[:,0] + np.array([np.random.rand() for _ in range(self.m)]) * (self.y_boundary[:,1]-self.y_boundary[:,0])
            
            self.x_nests[each_nest] = x_s.copy() # 写入一行基站的x坐标
            self.y_nests[each_nest] = y_s.copy()
        return self.x_nests,self.y_nests
    
    def calc_fitness(self, x_nests, y_nests):
        """
        计算适应度,即没有被一组基站的信号cover住的小区
        如果一个小区对于最近的基站距离大于10km,则适应度+1
        ---------------------------------------------------
        Input parameters:
            nests:  Nests' locations n*m array
        Output:
            Every nest's fitness
        """
        fitness = self.fitness.copy()
        for each_nest in range(self.n):
            # 把x,y组合成列表
            x = [[i] for i in x_nests[each_nest,:]]
            y = [[i] for i in y_nests[each_nest,:]]
            nest = np.concatenate((x, y), axis=1)
            # print(nest[:5])

            # 计算距离
            station2build = scipy.spatial.distance.cdist(nest,multi_neig_point,metric='euclidean')
            min_dis = station2build.min(0)    # 返回每列最小值
            fitness[0,each_nest] = min_dis[min_dis>10].size
            # TODO 为什么首次fitness就会有0值？
        # self.save_best(fitness)
        return fitness
        
    def save_best(self,fitness):
        if fitness.min() < self.best_fit:
            best_index = np.argmin(fitness)
            self.best_station_x = self.x_nests[best_index,:].copy()
            self.best_station_y = self.y_nests[best_index,:].copy()
            self.best_fit = fitness.min()

    def update_nests(self, step_coefficient=0.5):
        """
        This function is to get new nests' locations and use new better one to replace the old nest
        ---------------------------------------------------
        Input parameters:
            step_coefficient:  Step size scaling factor related to the problem's scale (default: 0.5)
        更新:self.x/y_nests
        """
        # generate steps using levy flight
        steps = self.levy_flight()
        x_nests = self.x_nests.copy()
        y_nests = self.y_nests.copy()

        for each_nest in range(self.n):
            # coefficient 0.01 is to avoid levy flights becoming too aggresive
            # and (nest[each_nest] - best_nest) could let the best nest be remained
            step_size_x = step_coefficient * steps[each_nest] * (x_nests[each_nest] - self.best_station_x)
            step_size_y = step_coefficient * steps[each_nest] * (y_nests[each_nest] - self.best_station_y)
            step_direction = np.random.rand(self.m)
            x_nests[each_nest] += step_size_x * step_direction
            y_nests[each_nest] += step_size_y * step_direction

            # apply boundary condtions
            x_nests[each_nest][x_nests[each_nest] < self.x_boundary[0,0]] = self.x_boundary[0,0]
            x_nests[each_nest][x_nests[each_nest] > self.x_boundary[0,1]] = self.x_boundary[0,1]
            y_nests[each_nest][y_nests[each_nest] < self.y_boundary[0,0]] = self.y_boundary[0,0]
            y_nests[each_nest][y_nests[each_nest] > self.y_boundary[0,1]] = self.y_boundary[0,1]

        temp_fitness = self.fitness.copy()
        new_fitness = self.calc_fitness(x_nests, y_nests)
        self.x_nests[list(new_fitness > temp_fitness)][:] = x_nests[list(new_fitness > temp_fitness)]
        self.y_nests[list(new_fitness > temp_fitness)] = y_nests[list(new_fitness > temp_fitness)]
        self.save_best(self.calc_fitness(self.x_nests, self.y_nests))
        
    def levy_flight(self, beta = 0.5):
        """
        This function implements Levy's flight.
        ---------------------------------------------------
        Input parameters:
            n: Number of steps 
            m: Number of dimensions
            beta: Power law index (note: 1 < beta < 2)
        Output:
            'n' levy steps in 'm' dimension
        """
        sigma_u = (sc_special.gamma(1+beta)*np.sin(np.pi*beta/2)/(sc_special.gamma((1+beta)/2)*beta*(2**((beta-1)/2))))**(1/beta)
        sigma_v = 1

        u =  np.random.normal(0, sigma_u, (self.n, self.m))
        v = np.random.normal(0, sigma_v, (self.n, self.m))

        steps = u/((np.abs(v))**(1/beta))
        return steps

    # def abandon_nests(self):
    #     """
    #     Some cuckoos' eggs are found by hosts, and are abandoned.So cuckoos need to find new nests.
    #     ---------------------------------------------------
    #     Input parameters:
    #         nests: Current nests' locations
    #         lower_boundary: Lower bounary (example: lower_boundary = (-2, -2, -2))
    #         upper_boundary: Upper boundary (example: upper_boundary = (2, 2, 2))
    #         pa: Possibility that hosts find cuckoos' eggs
    #     更新: self.x/y_nests
    #     """
    #     x_nests = self.x_nests
    #     y_nests = self.y_nests
    #     for each_nest in range(self.n):
    #         if (np.random.rand() < self.pa):
    #             # 丢弃x、y现在的位置,移动到新的地区
    #             x_step_size = np.random.rand() * (x_nests[np.random.randint(0, self.n)] - x_nests[np.random.randint(0, self.n)])
    #             y_step_size = np.random.rand() * (y_nests[np.random.randint(0, self.n)] - y_nests[np.random.randint(0, self.n)])
    #             x_nests[each_nest] += x_step_size
    #             y_nests[each_nest] += y_step_size

    #             # apply boundary condtions
    #             x_nests[each_nest][x_nests[each_nest] < self.x_boundary[0,0]] = self.x_boundary[0,0]
    #             x_nests[each_nest][x_nests[each_nest] > self.x_boundary[0,1]] = self.x_boundary[0,1]
    #             y_nests[each_nest][y_nests[each_nest] < self.y_boundary[0,0]] = self.y_boundary[0,0]
    #             y_nests[each_nest][y_nests[each_nest] > self.y_boundary[0,1]] = self.y_boundary[0,1]
    #     self.calc_fitness(x_nests, y_nests)
    
    def run(self):
        self.generate_nests()
        self.fitness = self.calc_fitness(self.x_nests,self.y_nests)
        for _ in range(self.iter_num):
            self.update_nests()
            # self.abandon_nests()
            self.fitness = self.calc_fitness(self.x_nests,self.y_nests)
            self.save_best(self.fitness)
            print(self.best_fit,end='\t')

    def draw(self, station_x, station_y):
        """
        In: 一组基站的x,y
        Out: 画基站与小区
        """
        plt.scatter(self.buildings[:,0], self.buildings[:,1], s =10, c = 'b', alpha=0.6)
        plt.scatter(station_x, station_y, s = 10, c = 'orange', alpha = 1)
        plt.show()
    
"""
用布谷鸟算法寻优:
设每代50组数据一并计算, 起始迭代的基站数m为小区/2, 
适应性函数为未被包含的小区数, 适应性函数为0时, 基站数m-1进行新的布谷鸟搜索
所以当前布谷鸟跳出的条件是: 循环次数达到或者适应度函数为0
取到最优的情况是适应度函数不为零,但迭代次数用完
"""
# 取出数据
df_lnla = pd.read_excel('深圳市楼盘带经纬度.xls')
df_lnla.head()
build_num = len(df_lnla['小区'])

# 转换成km
df_km = df_lnla.copy()
df_km['经度'] *= 111.1
df_km['纬度'] *= 92
df_km.columns = ['小区','x','y']
df_km.head()

# 画出来吧
x = np.array(df_km['x'])
y = np.array(df_km['y'])
assert(len(x) == len(y))
plt.scatter(x,y,s=10,c='r',alpha=0.7)

# 生成一个元素存放一个点的数组
x = [[i] for i in x]
y = [[i] for i in y]
buildings = np.concatenate((x, y), axis=1)

# 计算各小区间的的距离
build2build = scipy.spatial.distance.cdist(buildings,buildings,metric='euclidean') # 欧式距离算各点间的距离
# 输出到excel中
df_build2build = pd.DataFrame(build2build)
df_build2build.columns = df_km['小区']
df_build2build.index = df_km['小区']
df_build2build.to_csv('小区与小区间的距离.csv', encoding="utf_8_sig")
df_build2build.head()

# 找出各小区与其距离20km内的小区
close_ind = np.argwhere((build2build <=20) & (build2build != 0))
close_dict = {i:set([]) for i in range(build_num)}   # 以字典形式存起来
for _ in close_ind:
    close_dict[_[0]].add(_[1])

s = {0:[], 1:[], 2:[]}  # 存放已建立基站的km坐标
close_temp = close_dict
cover = set([]) # 信号已覆盖的小区
# proceing = set([])  #正在处理的多邻居小区
# neig_cluster = []   #   存放多小区的邻居与邻居的邻居

"""对0、1邻居小区的建基站,多小区的进行聚集合并"""
for k in list(close_temp.keys()):
    v = close_temp[k]

    # 处理0个小区的
    if len(v) == 0:
        #  print(k) 
        #   附近生成随机基站
        x_s = df_km.iloc[k, 1]+np.random.randint(-10, high=10,size=1)
        y_s = df_km.iloc[k, 2]+np.random.randint(-10, high=10,size=1)
        s[0].append([x_s,y_s])
        cover.add(k)
        del close_temp[k]

    #   处理1个小区的
    elif len(v) == 1:
        if k not in cover:
            x1, y1 = df_km.iloc[k, 1], df_km.iloc[k, 2]
            x2, y2 = df_km.iloc[v, 1], df_km.iloc[v, 2]
            s[1].append([(x1 + x2) / 2, (y1 + y2) / 2])
            cover.update(k, v)
        else:pass
        del close_temp[k]

    #   聚集多小区的邻居与邻居的邻居
    else:
        # if k not in cover:
        #     temp = close_temp[k].add(k)
        #     for i in close_temp[k]:
        #         temp.update(close_temp[i],{i})
        #     temp
        pass

#   将多个小区新建表单存储
multi_neig = close_temp.keys()
lng_a, lat_a = df_km.iloc[list(multi_neig),1], df_km.iloc[list(multi_neig),2]
df_multi_neig = pd.DataFrame(index=multi_neig)
df_multi_neig['x'] = lng_a
df_multi_neig['y'] = lat_a
df_multi_neig.to_csv('多个邻居的小区位置.csv', encoding="utf_8_sig")
df_multi_neig.tail()

# 生成一个元素存放一个小区位置的数组
x = df_multi_neig['x']
y = df_multi_neig['y']
x = [[i] for i in df_multi_neig['x']]
y = [[i] for i in df_multi_neig['y']]
x_boundary = [min(x), max(x)]
y_boundary = [min(y), max(y)]
multi_neig_point = np.concatenate((np.array(x), np.array(y)), axis=1)
multi_neig_point[:5]


"""
用布谷鸟算法寻优:
设每代50组数据一并计算, 起始迭代的基站数m为小区/2, 
适应性函数为未被包含的小区数, 适应性函数为0时, 基站数m-1进行新的布谷鸟搜索
所以当前布谷鸟跳出的条件是: 循环次数达到或者适应度函数为0
取到最优的情况是适应度函数不为零,但迭代次数用完
"""

for m in range(10,1,-1):
    print('\n','='*30)
    print("对多邻居建设%d个基站"%m)
    cuckoo_station = cuckoo(50, m = m, buildings=multi_neig_point,iter_num=200, x_boundary=x_boundary, y_boundary=y_boundary, step = 2)
    cuckoo_station.run()
    cuckoo_station.draw(cuckoo_station.best_station_x, cuckoo_station.best_station_y)
    cuckoo_station.best_fit

    # 存储经纬度
    # df_build2build.columns.append('%d_x'%m)
    # df_build2build.columns.append('%d_y'%m)
    # df_multi_station = pd.DataFrame((cuckoo_station.best_station_x/111.1, cuckoo_station.best_station_y/92))
    df_multi_station = pd.DataFrame()
    df_multi_station.insert(loc=len(df_multi_station.columns), column=str('经度%d'%m), value=cuckoo_station.best_station_x/111.1)
    df_multi_station.insert(loc=len(df_multi_station.columns), column=str('纬度%d'%m), value=cuckoo_station.best_station_y/92)
    # df_build2build['x'] = cuckoo_station.best_station_x/111.1
    # df_build2build['y'] = cuckoo_station.best_station_y/92
    df_multi_station.to_csv('多小区%d基站位置.csv'%m, encoding="utf_8_sig")