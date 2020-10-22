from torch.utils.data import DataLoader
from dataloaders.datasets import stereo
import pdb

def make_data_loader(args, **kwargs):
        ############################ sceneflow ###########################
        if args.dataset == 'sceneflow':              
            trainA_list= 'dataloaders/lists/sceneflow_search_trainA.list' #randomly select 10,000 from the original training set
            trainB_list= 'dataloaders/lists/sceneflow_search_trainB.list' #randomly select 10,000 from the original training set
            val_list   = 'dataloaders/lists/sceneflow_search_val.list'   #randomly select 1,000 from the original training set
            train_list = 'dataloaders/lists/sceneflow_train.list'  #original training set: 35,454
            test_list  = 'dataloaders/lists/sceneflow_test.list'   #original test set:4,370
            trainA_set = stereo.DatasetFromList(args, trainA_list, [args.crop_height, args.crop_width], True)
            trainB_set = stereo.DatasetFromList(args, trainB_list, [args.crop_height, args.crop_width], True)
            train_set  = stereo.DatasetFromList(args, train_list, [args.crop_height, args.crop_width], True)
            val_set    = stereo.DatasetFromList(args, val_list,  [576,960], False)
            test_set   = stereo.DatasetFromList(args, test_list,  [576,960], False)

            if args.stage == 'search':
                train_loaderA = DataLoader(trainA_set, batch_size=args.batch_size, shuffle=True, **kwargs)
                train_loaderB = DataLoader(trainB_set, batch_size=args.batch_size, shuffle=True, **kwargs)
            elif args.stage == 'train':
                train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
            else:
                raise Exception('parameters not set properly')

            val_loader  = DataLoader(val_set, batch_size=args.testBatchSize, shuffle=False, **kwargs)
            test_loader = DataLoader(test_set, batch_size=args.testBatchSize, shuffle=False, **kwargs)

            if args.stage == 'search':
                return train_loaderA, train_loaderB, val_loader, test_loader 
            elif args.stage == 'train':
                return train_loader, test_loader

        ############################ kitti15 ###########################
        elif args.dataset == 'kitti15':              
            train_list= 'dataloaders/lists/kitti2015_train180.list'
            test_list = 'dataloaders/lists/kitti2015_val20.list'  
            train_set = stereo.DatasetFromList(args, train_list, [args.crop_height, args.crop_width], True)
            test_set  = stereo.DatasetFromList(args, test_list,  [384,1248], False)
           
            train_loader= DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
            test_loader = DataLoader(test_set, batch_size=args.testBatchSize, shuffle=False, **kwargs)
            return train_loader, test_loader

        ############################ kitti12 ###########################
        elif args.dataset == 'kitti12':              
            train_list= 'dataloaders/lists/kitti2012_train170.list'
            test_list = 'dataloaders/lists/kitti2012_val24.list'  
            train_set = stereo.DatasetFromList(args, train_list, [args.crop_height, args.crop_width], True)
            test_set  = stereo.DatasetFromList(args, test_list,  [384,1248], False)
           
            train_loader= DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
            test_loader = DataLoader(test_set, batch_size=args.testBatchSize, shuffle=False, **kwargs)
            return train_loader, test_loader

        ############################ middlebury ###########################
        elif args.dataset == 'middlebury':
            train_list= 'dataloaders/lists/middeval3_train.list'
            test_list = 'dataloaders/lists/middeval3_train.list'
            train_set = stereo.DatasetFromList(args, train_list, [args.crop_height, args.crop_width], True)
            test_set  = stereo.DatasetFromList(args, test_list,  [1008,1512], False)

            train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
            test_loader  = DataLoader(test_set, batch_size=args.testBatchSize, shuffle=False, **kwargs)
            return train_loader, test_loader
        else:
            raise NotImplementedError
