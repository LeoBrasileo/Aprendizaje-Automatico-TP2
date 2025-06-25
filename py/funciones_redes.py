import torch

def ejecutar_epoch_entrenamiento(model, dataloader, optimizer, criterion):
    device = next(model.parameters()).device
    model.train()
    loss_total = 0

    for batch in dataloader:
        optimizer.zero_grad()

        # Cargo los datos al device que estemos utlizando (caso de estar usando GPU)
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
            
        # Paso forward
        logits_punt_inic, logits_punt_final, logits_capitalizacion = model(batch['token_ids'])

        # Reshape de los logits para CE:
        logits_punt_inic = logits_punt_inic.reshape(-1, logits_punt_inic.size(-1))
        logits_punt_final = logits_punt_final.reshape(-1, logits_punt_final.size(-1))
        logits_capitalizacion = logits_capitalizacion.reshape(-1, logits_capitalizacion.size(-1))

        # Definiendo los targets (y reshape para CE):
        target_punt_inic = batch['puntuacion_inicial'].reshape(-1).long()
        target_punt_final = batch['puntuacion_final'].reshape(-1).long()
        target_capitalizacion = batch['capitalizacion'].reshape(-1).long()

        # Calculamos la loss como la suma de las 3 losses
        loss = criterion(logits_capitalizacion, target_capitalizacion) + criterion(logits_punt_inic, target_punt_inic) + criterion(logits_punt_final, target_punt_final)

        # Paso backward
        loss.backward()

        # Gradient clipping para evitar exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        loss_total += loss.item()


    return loss_total / len(dataloader)

def evaluar_modelo(model, dataloader, criterion, epoch_actual, cant_epochs, device):     # El parámetro epoch_actual es sólo con el fin de printear y ver resultados del modelo en ciertas epochs.
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)

            output_punt_inic, output_punt_final, output_capitalizacion = model(batch['token_ids'])

            # Reshape para usar en CE
            output_punt_inic = output_punt_inic.reshape(-1, output_punt_inic.size(-1))
            output_punt_final = output_punt_final.reshape(-1, output_punt_final.size(-1))
            output_capitalizacion = output_capitalizacion.reshape(-1, output_capitalizacion.size(-1))
            
            target_punt_inic = batch['puntuacion_inicial'].reshape(-1).long()
            target_punt_final = batch['puntuacion_final'].reshape(-1).long()
            target_capitalizacion = batch['capitalizacion'].reshape(-1).long()

            loss = criterion(output_punt_inic.float(), target_punt_inic) + criterion(output_punt_final.float(), target_punt_final) + criterion(output_capitalizacion.float(), target_capitalizacion)
            total_loss += loss.item()

            # Imprimir las predicciones y targets del primer batch para visualizarlas. El único fin es visualizarlo.
            if (epoch_actual + 1) % max(1, cant_epochs // 10) == 0 and batch_idx == 0:
                    pred_puntuacion_inicial = torch.argmax(output_punt_inic, dim=-1)  
                    pred_puntuacion_final = torch.argmax(output_punt_final, dim=-1)
                    pred_capitalizacion = torch.argmax(output_capitalizacion, dim=-1)

                    print("Predicción puntuación inicial:", pred_puntuacion_inicial.cpu().tolist())
                    print("Target puntuación inicial:   ", target_punt_inic.cpu().tolist())
                    print()
                    print("Predicción puntuación final:  ", pred_puntuacion_final.cpu().tolist())
                    print("Target puntuación final:      ", target_punt_final.cpu().tolist())
                    print()
                    print("Predicción capitalización:    ", pred_capitalizacion.cpu().tolist())
                    print("Target capitalización:        ", target_capitalizacion.cpu().tolist())
                    print("\n" + "-"*50 + "\n")
        
    return total_loss / len(dataloader)


def entrenar_modelo(modelo, datos_entrenamiento, datos_validacion, optimizador, criterio, cant_epochs, device='cpu'):
    train_losses = []
    val_losses = []
    print("Iniciando entrenamiento...")
    print("-" * 50)
    for epoch in range(cant_epochs):
        # Entrenamiento
        train_loss = ejecutar_epoch_entrenamiento(modelo, datos_entrenamiento, optimizador, criterio)
        train_losses.append(train_loss)
        
        # Validación
        val_loss = evaluar_modelo(modelo, datos_validacion, criterio, epoch, cant_epochs, device=device)
        val_losses.append(val_loss)
        
        if (epoch + 1) % max(1, cant_epochs // 10) == 0:
            print(f'Época {epoch+1}/{cant_epochs}')
            print(f'  Pérdida Entrenamiento: {train_loss:.4f}')
            print(f'  Pérdida Validación: {val_loss:.4f}')
            print(f'  {"Mejorando" if val_loss < min(val_losses[:-1] + [float("inf")]) else "Empeorando"}')

    print("Entrenamiento completado!")
    # agrego el return para hacer curva de aprendizaje
    return train_losses, val_losses 