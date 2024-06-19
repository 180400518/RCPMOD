def get_default_config(data_name):
    
    if data_name in ['Scene_15']:
        """The default configs."""
        return dict(
            Autoencoder=dict(
                arch1=[20,1024,1024,64],
                arch2=[59,1024,1024,64],
                activations='relu',
                batchnorm=False,
                latent_dim=64,
            ),
            training=dict(
                seed=0,
                missing_rate=[0.3],
                epoch=[90],
                attr_outlier_rate=[0.02,0.05,0.08],
                class_outlier_rate=[0.02,0.05,0.08],
                attr_class_outlier_rate=[0.02,0.05,0.08],
                loss_paras=[1,1,0.4],
                lr=[1e-3],
                start_seed=[1],
                wnn_k=[7],
                Rw=0.25,
                start_Keleo_loss_epoch=0,
                saveTnse_epoch=50,
                change_neg_epoch=5,
                Contr_on_impu_Stepoch=50,
                
                Using_K_Contr=1,
                
                Using_KeLeo=1,
                Using_Contr=True,
                Contr_on_Impu=True,
                
                
                Moreratio=0.05,
                Memorysize=5,
                Using_schedule=1,
                warmup_max=0.02,
                schedule_epoch=400,
                
                dataset_name='SCENE15',    
                Run_name="",
                
                Pred_use_latent=True,
                batch_size=[256],
                num_neibs=[6],   
                
                Change_Knn=True,
                change_knn_epoch=50,
                start_wnn=0,
                change_KmeansEntropy=50,
                
            ),
        )

    
    elif data_name in ['BDGP']:
        """The default configs."""
        return dict(
            Autoencoder=dict(
                arch1=[1750,1024,64],
                arch2=[79,1024,64],
                activations1='relu',
                activations2='relu',
                batchnorm=False,
                latent_dim=64,
            ),
            training=dict(
                seed=0,
                missing_rate=[0.3],
                epoch=[130],
                attr_outlier_rate=[0.02,0.05,0.08],
                class_outlier_rate=[0.02,0.05,0.08],
                attr_class_outlier_rate=[0.02,0.05,0.08],
                loss_paras=[1,1,0.2],
                batch_size=[256],
                num_neibs=[6],
                Rw=0.25,
                lr=[1e-3],
 
                start_Keleo_loss_epoch=0,
                change_neg_epoch=1,
                Contr_on_impu_Stepoch=50,
                Save_Keleo_epoch=50,
                saveScore_epoch=25,
                
                Using_K_Contr=True,
                Using_KeLeo=1,
                Using_Contr=True,
                
                Moreratio=0.05,
                Memorysize=10,
                Using_schedule=True,
                warmup_max=0.01,
                schedule_epoch=1000,
                


                Save_Keleo_log=True,
                
                dataset_name='BDGP',    
                Run_name="",
                
                Pred_use_latent=True,   
                Contr_on_Impu=True,

                start_seed=[1],
                Change_Knn=True,
                change_knn_epoch=50,
                comp_epoch=0,
                
            ),
        )
    
    elif data_name in ['Fashion']:
        return dict(
            Autoencoder=dict(
                arch1=[784,1024,256],
                arch2=[784,1024,256],
                activations1='relu',
                activations2='relu',
                batchnorm=False,
                latent_dim=256,
            ),
            training=dict(
                
                seed=0,
                missing_rate=[0.3],
                epoch=[400],
                attr_outlier_rate=[0.02,0.05,0.08],
                class_outlier_rate=[0.02,0.05,0.08],
                attr_class_outlier_rate=[0.02,0.05,0.08],
                loss_paras=[1,1,0.4],
                lr=[1e-3],
                batch_size=[256],
                num_neibs=[6],
                Rw=0.5,

 
                start_Keleo_loss_epoch=0,
                change_neg_epoch=10,
                Contr_on_impu_Stepoch=50,
                
                Using_K_Contr=True,
                Using_KeLeo=1,
                Using_Contr=True,
                
                Moreratio=0.05,
                Memorysize=10,
                Using_schedule=True,
                warmup_max=0.05,
                schedule_epoch=700,



                Save_Keleo_log=True,
                
                dataset_name='Fashion',    
                Run_name="",
                
                Pred_use_latent=True,  
                Contr_on_Impu=True,
                start_seed=[1],
                Change_Knn=True,
                change_knn_epoch=50,
            ),
        )


    elif data_name in ['LandUse_21']:
        """The default configs."""
        return dict(
            Autoencoder=dict(
                arch1=[59,1024,1024,64],
                arch2=[40,1024,1024,64],
                activations='relu',
                batchnorm=False,
                latent_dim=64,
            ),
            training=dict(  
                seed=0,
                missing_rate=[0.3],
                epoch=[115],
                attr_outlier_rate=[0.02,0.05,0.08],
                class_outlier_rate=[0.02,0.05,0.08],
                attr_class_outlier_rate=[0.02,0.05,0.08],
                loss_paras=[1,1,0.2],
                lr=[1e-3],
                start_seed=[1],
                Rw=0.6,


                start_Keleo_loss_epoch=0,
                change_neg_epoch=5,
                Contr_on_impu_Stepoch=50,
                Using_K_Contr=True,
                Using_KeLeo=1,
                Using_Contr=True,
                
                Moreratio=0.05,
                Memorysize=5,
                Using_schedule=True,
                warmup_max=0.02,
                schedule_epoch=1000,

                Save_Keleo_log=True,
                
                dataset_name='LandUse21',    
                Run_name="",
                
                Pred_use_latent=True,
                start_dual_prediction=[100],
                batch_size=[256],
                num_neibs=[6],   
                Contr_on_Impu=True,
                Change_Knn=True,
                change_knn_epoch=50,
            ),
        )
    
    elif data_name in ['Handwritten']:
        """The default configs."""
        return dict(
            Autoencoder=dict(
                arch1=[240,1024,1024,64],
                arch2=[76,1024,1024,64],
                arch3=[216,1024,1024,64],
                arch4=[47,1024,1024,64],
                arch5=[64,1024,1024,64],
                arch6=[6,1024,1024,64],
                activations='relu',
                batchnorm=False,
                latent_dim=64,
            ),
            training=dict(
                
                seed=0,
                missing_rate=[0.3],
                epoch=[50],
                attr_outlier_rate=[0.02,0.05,0.08],
                class_outlier_rate=[0.02,0.05,0.08],
                attr_class_outlier_rate=[0.02,0.05,0.08],
                loss_paras=[1,1,0.2],
                lr=[1e-3],
                start_seed=[1],
                Rw=0.7,


                start_Keleo_loss_epoch=0,
                change_neg_epoch=5,
                Contr_on_impu_Stepoch=50,
                
                Using_K_Contr=True,
                
                Using_KeLeo=1,
                Using_Contr=True,
                Moreratio=0.05,
                Memorysize=5,
                Using_schedule=True,
                warmup_max=0.02,
                schedule_epoch=1000,


                Save_Keleo_log=True,
                
                dataset_name='Handwritten',    
                Run_name="",
                
                Pred_use_latent=True,
                batch_size=[256],
                num_neibs=[6],      
                Contr_on_Impu=True,
                Change_Knn=True,
                change_knn_epoch=50,
            ),
        )
    else:
        raise Exception('Undefined data_name')
