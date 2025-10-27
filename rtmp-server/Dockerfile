FROM nginx:1.23-alpine

# Instalar módulo RTMP
RUN apk add --no-cache nginx-mod-rtmp

# Copiar configuración
COPY nginx.conf /etc/nginx/nginx.conf

# Puerto para RTMP
EXPOSE 1935
# Puerto para estadísticas HTTP
EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
