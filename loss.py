def compute_loss_nm():
    # gen dict{label:[cord]}
    label_dict = {}
    im_target_HP = im_target.reshape((im.shape[0], im.shape[1]))
    for y in range(im.shape[0]):
        for x in range(im.shape[1]):
            label = im_target_HP[y][x]
            if label not in label_dict.keys():
                label_dict[label] = []
            label_dict[label].append((x, y))
    nLabels = len(np.unique(im_target))
    # random sample dict
    print('sample N group 3 point from label_dict')
    N = 5
    sample_dict = {}
    label_norm_list = []
    for label_list in label_dict.items():
        if label_list[0] not in sample_dict.keys():
            sample_dict[label_list[0]] = []
        try:
            list = random.sample(label_list[1], N*3)
        except:
            continue
        random.shuffle(list) 
        n = 5
        m = int(len(list)/n)
        list2 = []
        for i in range(0, len(list), m):
            list2.append(list[i:i+m])
        # find depth for selected points
        N_normal_group = []
        for group in list2:
            group_3d_point = []
            for p in group:
                val_depth = depth/5000
                d = val_depth[p[1]][p[0]]
                X = (p[0] - cx_d)/fx_d*d
                Y = (p[1] - cy_d)/fy_d*d
                group_3d_point.append(np.array([X, Y, d]))
            p12 = np.subtract(group_3d_point[1], group_3d_point[0])
            p13 = np.subtract(group_3d_point[2], group_3d_point[0])
            n = np.cross(p12, p13)
            N_normal_group.append(n)
        N_normal_group = itertools.permutations(N_normal_group, 2)
        cos_theta_list = []
        for normal_2 in N_normal_group:
            (x1, y1, z1), (x2, y2, z2) = normal_2
            cos_theta = (x1*x2+y1*y2+z1*z2)/(math.sqrt(x1*x1+y1*y1+z1*z1)*math.sqrt(x2*x2+y2*y2+z2*z2))
            cos_theta_list.append(cos_theta)
        normal_theta = np.var(cos_theta_list)
        # print("label {} normal var is {}".format(label_list[0], normal_theta))
        label_norm_list.append(normal_theta)
    print("{} labels are computed, var large {:.4f} | small {:.4f} | median {:.4f}".format(len(label_norm_list), np.max(label_norm_list), np.min(label_norm_list), np.median(label_norm_list)))
    loss_nm = loss_normal(np.percentile(label_norm_list, 30), loss_nm_target)
    print("normal loss is ", loss_nm)