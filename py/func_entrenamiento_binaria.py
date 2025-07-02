import torch

def padding(x, pad_index):
  res = x
  if x == pad_index:
    res = -100
  return res

def criterion_batch_sequence(batch_logits, batch_targets, criterion, task_offset, class_size):
  loss = 0
  batch_size = batch_targets.shape[0]
  for idx in range(batch_targets.shape[1]):
      loss += criterion(batch_logits[:, idx, :], batch_targets[:, idx]).sum()
  return loss

def ejecutar_epoch_entrenamiento(model, dataloader, optimizer, criterion):
    device = next(model.parameters()).device
    model.train()
    loss_total = 0
    loss_punt_inic_total = 0
    loss_punt_final_total = 0
    loss_capitalizacion_total = 0

    for batch in dataloader:

        # Cargo los datos al device que estemos utlizando (caso de estar usando GPU)
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
            
        # Paso forward
        batch['token_ids'] = batch['token_ids'].to(device)
        
        logits_clasifiacion = model(batch['token_ids'])

        # Definiendo los targets (y reshape para CE):
        target_punt_inic = batch['puntuacion_inicial'].long()#.reshape(-1)
        target_punt_final = batch['puntuacion_final'].long()#.reshape(-1)
        target_capitalizacion = batch['capitalizacion'].long()#.reshape(-1)

        # Calculamos la loss como la suma de las 3 losses
        loss_punt_inic = 0
        loss_punt_final = 0
        loss_capitalizacion = 0
        batch_size = target_punt_inic.shape[0]
        for idx in range(target_punt_inic.shape[1]):
            loss_punt_inic += criterion(logits_clasifiacion[:, idx, 0:2], target_punt_inic[:, idx]).sum()
            loss_punt_final += criterion(logits_clasifiacion[:, idx, 2:6], target_punt_final[:, idx]).sum()
            loss_capitalizacion += criterion(logits_clasifiacion[:, idx, 6:10], target_capitalizacion[:, idx]).sum()

        loss = (loss_punt_inic + loss_punt_final + loss_capitalizacion)

        optimizer.zero_grad()
        # Paso backward
        loss.backward()

        # Gradient clipping para evitar exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        loss_total += loss.item()

        loss_punt_inic_total += loss_punt_inic.item() 
        loss_punt_final_total += loss_punt_final.item()
        loss_capitalizacion_total += loss_capitalizacion.item()


    return loss_total / len(dataloader), loss_punt_inic_total / len(dataloader), loss_punt_final_total / len(dataloader), loss_capitalizacion_total / len(dataloader)

def evaluar_modelo(model, dataloader, criterion, epoch_actual, cant_epochs, device):     # El parámetro epoch_actual es sólo con el fin de printear y ver resultados del modelo en ciertas epochs.
    model.eval()
    loss_total = 0
    loss_punt_inic_total = 0
    loss_punt_final_total = 0
    loss_capitalizacion_total = 0

    device = next(model.parameters()).device
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            batch['token_ids'] = batch['token_ids'].to(device)
            logits_clasifiacion = model(batch['token_ids'])
            print(output_capitalizacion.shape)
            print(batch['capitalizacion'].shape)
            
            target_punt_inic = batch['puntuacion_inicial'].long()#.reshape(-1)
            target_punt_final = batch['puntuacion_final'].long()#.reshape(-1)
            target_capitalizacion = batch['capitalizacion'].long()#.reshape(-1)

            logits_punt_inicial = logits_clasifiacion[:, :, 0:2]
            logits_punt_final = logits_clasifiacion[:, :, 2:6]
            logits_cap = logits_clasifiacion[:, :, 6:10]

            # Calculamos la loss como la suma de las 3 losses
            loss_punt_inic = 0
            loss_punt_final = 0
            loss_capitalizacion = 0
            batch_size = target_punt_inic.shape[0]
            for idx in range(target_punt_inic.shape[1]):
                loss_punt_inic += criterion(logits_punt_inicial[:, idx, :], target_punt_inic[:, idx]).sum()
                loss_punt_final += criterion(logits_punt_final[:, idx, :], target_punt_final[:, idx]).sum()
                loss_capitalizacion += criterion(logits_cap[:, idx, :], target_capitalizacion[:, idx]).sum()
            loss = (loss_punt_inic + loss_punt_final + loss_capitalizacion)

            loss_total += loss.item()
            loss_punt_inic_total += loss_punt_inic.item()
            loss_punt_final_total += loss_punt_final.item()
            loss_capitalizacion_total += loss_capitalizacion.item()


            # Imprimir las predicciones y targets del primer batch para visualizarlas. El único fin es visualizarlo.
            if (epoch_actual + 1) % max(1, cant_epochs // 10) == 0 and batch_idx == 0:
                    pred_puntuacion_inicial, pred_puntuacion_final, pred_capitalizacion = torch.Tensor()
                    for i in range(logits_clasificacion.shape[0]) 
                        pred_puntuacion_inicial.append( torch.argmax(logits_punt_inicial[i], dim=-1) ) 
                        pred_puntuacion_final.append( torch.argmax(logits_punt_final[i], dim=-1) )
                        pred_capitalizacion.append( torch.argmax(logits_cap[i], dim=-1) )

                    print("Predicción puntuación inicial:", pred_puntuacion_inicial.cpu().tolist())
                    print("Target puntuación inicial:   ", target_punt_inic.cpu().tolist())
                    print()
                    print("Predicción puntuación final:  ", pred_puntuacion_final.cpu().tolist())
                    print("Target puntuación final:      ", target_punt_final.cpu().tolist())
                    print()
                    print("Predicción capitalización:    ", pred_capitalizacion.cpu().tolist())
                    print("Target capitalización:        ", target_capitalizacion.cpu().tolist())
                    print("\n" + "-"*50 + "\n")
        
    return loss_total / len(dataloader), loss_punt_inic_total / len(dataloader), loss_punt_final_total / len(dataloader), loss_capitalizacion_total / len(dataloader)



def entrenar_modelo(modelo, datos_entrenamiento, datos_validacion, optimizador, criterio, cant_epochs, device='cpu'):
    train_losses = []
    val_losses = []
    punt_inicial_train_losses, punt_final_train_losses, punt_cap_train_losses = [], [], []
    punt_inicial_val_losses, punt_final_val_losses, punt_cap_val_losses = [], [], []

    print("Iniciando entrenamiento...")
    print("-" * 50)
    for epoch in range(cant_epochs):
        # Entrenamiento
        train_loss, punt_inicial_train_loss, punt_final_train_loss, punt_cap_train_loss = ejecutar_epoch_entrenamiento(modelo, datos_entrenamiento, optimizador, criterio)

        train_losses.append(train_loss)
        punt_inicial_train_losses.append(punt_inicial_train_loss)
        punt_final_train_losses.append(punt_final_train_loss)
        punt_cap_train_losses.append(punt_cap_train_loss)

        # Validación
        val_loss, punt_inicial_val_loss, punt_final_val_loss, punt_cap_val_loss = evaluar_modelo(modelo, datos_validacion, criterio, epoch, cant_epochs, device=device)
        
        punt_inicial_val_losses.append(punt_inicial_val_loss)
        punt_final_val_losses.append(punt_final_val_loss)
        punt_cap_val_losses.append(punt_cap_val_loss)
        val_losses.append(val_loss)
        
        if (epoch + 1) % max(1, cant_epochs // 10) == 0:
            print(f'Época {epoch+1}/{cant_epochs}')
            print(f'  Pérdida Entrenamiento: {train_loss:.4f}')
            print(f'  Pérdida Validación: {val_loss:.4f}')
            print(f'  {"Mejorando" if val_loss < min(val_losses[:-1] + [float("inf")]) else "Empeorando"}')

    print("Entrenamiento completado!")
    # agrego el return para hacer curva de aprendizaje
    return train_losses, val_losses, punt_inicial_train_losses, punt_final_train_losses, punt_cap_train_losses, punt_inicial_val_losses, punt_final_val_losses, punt_cap_val_losses 
